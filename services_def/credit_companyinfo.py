
from typing import List
from sqlalchemy.orm import Session
from database import SessionLocal
from models.credit_models import ReportContent

        
def get_autocomplete_suggestions(query: str) -> List[str]:
    db: Session = SessionLocal()
    if query:
        results = db.query(ReportContent.corp_name).filter(ReportContent.corp_name.like(f"{query}%")).limit(5).all()
    else:
        results = []

    return [result[0] for result in results]