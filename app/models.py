from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base  
from pgvector.sqlalchemy import Vector

class Ticket(Base):
    __tablename__ = "ticket"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(100) ,default="open")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Classification(Base):
    __tablename__ = "classification"

    id: Mapped[int] = mapped_column(primary_key=True) 
    ticket_id: Mapped[int] = mapped_column(ForeignKey("ticket.id"))
    category: Mapped[str] = mapped_column(String(100))
    urgency: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column()

class Embedding(Base):
    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("ticket.id"))
    vector: Mapped[list[float]] = mapped_column(Vector(768))
    