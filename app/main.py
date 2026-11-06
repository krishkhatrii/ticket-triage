from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import init_db, get_db
from app.llm import classify_ticket, embed
from app.schemas import Classification, SourceOut, TicketCreate, TicketOut, TriageOut
from app.models import Ticket, Classification as ClassificationRow
from app.retrieval import find_similar
from sqlalchemy import select

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Ticket Triage API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/tickets", response_model=TriageOut, status_code=201)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    classification = classify_ticket(payload.subject, payload.body)
    vector = embed(payload.subject + "\n" + payload.body)
    rows = find_similar(db, vector, k=3)

    ticket = Ticket(subject=payload.subject, body=payload.body)
    db.add(ticket)
    db.flush() 
    db.add(ClassificationRow(
        ticket_id=ticket.id,
        category=classification.category.value,
        urgency=classification.urgency.value,
        confidence=classification.confidence,
    ))
    db.commit()
    db.refresh(ticket) 

    sources = [
        SourceOut(
            ticket_id=r.Ticket.id,
            subject=r.Ticket.subject,
            answer=r.Ticket.answer,
            similarity=round(1 - r.distance, 3),
        )
        for r in rows
    ]
    return TriageOut(
        ticket=TicketOut.model_validate(ticket),
        classification=classification,
        sources=sources,
    )

@app.get("/tickets/{ticket_id}", response_model = TicketOut)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket

@app.get("/tickets", response_model = list[TicketOut])
def list_tickets(status: str |None = None, db: Session = Depends(get_db)):
    query = select(Ticket)
    if status is not None:
        query = query.where(Ticket.status == status)
    return db.scalars(query).all()
