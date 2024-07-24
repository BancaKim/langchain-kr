
from typing import List
from database import SessionLocal
from sqlalchemy.orm import Session
from models.credit_models import ReportContent


def get_corp_info_code(corp_code: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(ReportContent).filter(ReportContent.corp_code == corp_code).first()
        return selectedcompany
    finally:
        db.close()
        
def get_corp_info_name(corp_name: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(ReportContent).filter(ReportContent.corp_name == corp_name).first()
        return selectedcompany
    finally:
        db.close()
        
def get_autocomplete_suggestions(search_type: str, query: str) -> List[str]:
    db: Session = SessionLocal()
    if search_type == "corp_name":
        results = db.query(ReportContent.corp_name).filter(ReportContent.corp_name.like(f"{query}%")).limit(5).all()
    elif search_type == "corp_code":
        results = db.query(ReportContent.corp_code).filter(ReportContent.corp_code.like(f"{query}%")).limit(5).all()
    else:
        results = []

    return [result[0] for result in results]