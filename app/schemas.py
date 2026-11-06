from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

class TicketCreate(BaseModel):
    subject : str
    body : str

class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    body: str
    status: str
    created_at: datetime

class Category(str, Enum):
    general_inquiry = "General Inquiry"
    sales  = "Sales and Pre-Sales"
    it_support = "IT Support"
    product_support = "Product Support"
    returns = "Returns and Exchanges"
    outages = "Service Outages and Maintenance"
    customer_service = "Customer Service"
    hr = "Human Resources"
    billing = "Billing and Payments"
    technical_support = "Technical Support"

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class Classification(BaseModel):
    category: Category
    urgency : Urgency
    confidence : float= Field(ge=0, le=1)

class SourceOut(BaseModel):
    ticket_id: int
    subject: str
    answer: str | None
    similarity: float

class TriageOut(BaseModel):
    ticket: TicketOut
    classification: Classification
    sources: list[SourceOut]