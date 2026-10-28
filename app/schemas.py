from datetime import datetime
from pydantic import BaseModel, ConfigDict

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
