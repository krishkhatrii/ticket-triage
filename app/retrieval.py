from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Embedding, Ticket

def find_similar(db: Session, vector: list[float], k: int=3, exclude_id: int | None = None):
    distance = Embedding.vector.cosine_distance(vector)

    query = (
        select(Ticket, distance.label("distance"))
        .join(Embedding, Embedding.ticket_id == Ticket.id)
        .where(Ticket.status == "resolved")
        .order_by(distance)
        .limit(k)
    )
    if exclude_id is not None:
        query = query.where(Ticket.id != exclude_id)
    return db.execute(query).all()