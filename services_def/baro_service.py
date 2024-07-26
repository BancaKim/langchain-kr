
from typing import List
from database import SessionLocal
from sqlalchemy import func, cast, Integer
from sqlalchemy.orm import Session
from sqlalchemy import text
from models.baro_models import CompanyInfo, FS2023, FS2022, FS2021, FS2020, StockData

def search_company(db: Session, keyword: str) -> List[str]:

    keyword_pattern = f"%{keyword}%"
    
    query = text("""
        SELECT jurir_no 
        FROM companyInfo 
        WHERE corp_name LIKE :keyword_pattern 
        OR jurir_no = :keyword 
        OR bizr_no = :keyword
    """)
    result = db.execute(query, {"keyword_pattern": keyword_pattern, "keyword": keyword})
    jurir_nos = [row.jurir_no for row in result.fetchall()]
    print(f"Found jurir_no: {jurir_nos}")  # 터미널에 출력
    return jurir_nos

def get_company_infoFS_list(db: Session, jurir_no_list: List[str]):
    query = db.query(
        CompanyInfo.corp_code,
        CompanyInfo.corp_name,
        CompanyInfo.corp_name_eng,
        CompanyInfo.stock_name,
        CompanyInfo.stock_code,
        CompanyInfo.ceo_nm,
        CompanyInfo.corp_cls,
        CompanyInfo.jurir_no.label("company_jurir_no"),
        CompanyInfo.bizr_no,
        CompanyInfo.adres,
        CompanyInfo.hm_url,
        CompanyInfo.ir_url,
        CompanyInfo.phn_no,
        CompanyInfo.fax_no,
        CompanyInfo.induty_code,
        CompanyInfo.est_dt,
        CompanyInfo.acc_mt,
        FS2023.id,
        FS2023.baseDate,
        FS2023.bizYear,
        FS2023.currency,
        FS2023.fsCode,
        FS2023.fsName,
        cast(FS2023.totalAsset2023 / 100000000, Integer).label('totalAsset2023'),
        cast(FS2023.totalDebt2023 / 100000000, Integer).label('totalDebt2023'),
        cast(FS2023.totalEquity2023 / 100000000, Integer).label('totalEquity2023'),
        cast(FS2023.capital2023 / 100000000, Integer).label('capital2023'),
        cast(FS2023.revenue2023 / 100000000, Integer).label('revenue2023'),
        cast(FS2023.operatingIncome2023 / 100000000, Integer).label('operatingIncome2023'),
        cast(FS2023.earningBeforeTax2023 / 100000000, Integer).label('earningBeforeTax2023'),
        cast(FS2023.netIncome2023 / 100000000, Integer).label('netIncome2023'),
        FS2023.debtRatio2023,
        FS2023.margin2023,
        FS2023.turnover2023,
        FS2023.leverage2023,
        FS2023.created_at
    ).join(FS2023, CompanyInfo.jurir_no == FS2023.jurir_no).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()
    
    return query

def get_company_info_list(db: Session, jurir_no_list: List[str]) -> List[CompanyInfo]:
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()



def get_company_info_list(db: Session, jurir_no_list: List[str]) -> List[CompanyInfo]:
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()

def get_company_info(db: Session, jurir_no: str):
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()

def get_Stock_data(db: Session, corp_code: str):
    return db.query(StockData).filter(StockData.corp_code == corp_code).first()

def get_FS2023_list(db: Session, jurir_no_list: List[str]) -> List[FS2023]:
    return db.query(FS2023).filter(FS2023.jurir_no.in_(jurir_no_list)).all()

def get_FS2023(db: Session, jurir_no: str):
    return db.query(FS2023).filter(FS2023.jurir_no == jurir_no).first()

def get_FS2022(db: Session, jurir_no: str):
    return db.query(FS2022).filter(FS2022.jurir_no == jurir_no).first()

def get_FS2021(db: Session, jurir_no: str):
    return db.query(FS2021).filter(FS2021.jurir_no == jurir_no).first()

def get_FS2020(db: Session, jurir_no: str):
    return db.query(FS2020).filter(FS2020.jurir_no == jurir_no).first()
    

def get_corp_info_code(corp_code: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.corp_code == corp_code).first()
        return selectedcompany
    finally:
        db.close()
        
def get_corp_info_name(corp_name: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.corp_name == corp_name).first()
        return selectedcompany
    finally:
        db.close()
        
def get_corp_info_jurir_no(jurir_no: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
        return selectedcompany
    finally:
        db.close()
        
        
def get_autocomplete_suggestions(search_type: str, query: str) -> List[str]:
    db: Session = SessionLocal()
    if search_type == "corp_name":
        results = db.query(CompanyInfo.corp_name).filter(CompanyInfo.corp_name.like(f"{query}%")).limit(5).all()
    elif search_type == "jurir_no":
        results = db.query(CompanyInfo.jurir_no).filter(CompanyInfo.jurir_no.like(f"{query}%")).limit(5).all()
    elif search_type == "corp_code":
        results = db.query(CompanyInfo.corp_code).filter(CompanyInfo.corp_code.like(f"{query}%")).limit(5).all()
    else:
        results = []

    return [result[0] for result in results]