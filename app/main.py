from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import init_db, get_db
from app.schemas import TicketCreate, TicketOut
from app.models import Ticket
from sqlalchemy import select

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Ticket Triage API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/tickets", response_model=TicketOut, status_code=201)
def ticket_create(payload: TicketCreate, db: Session = Depends(get_db)):
    ticket = Ticket(subject = payload.subject, body = payload.body)
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket 

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
