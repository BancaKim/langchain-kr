import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Union
from bs4 import BeautifulSoup
from fastapi import requests
import httpx
from jinja2 import Template
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


def get_company_info2(db: Session, corp_code: str):
    return db.query(CompanyInfo).filter(CompanyInfo.corp_code == corp_code).first()

def get_Stock_data(db: Session, corp_code: str):
    stock_data = db.query(StockData).filter(StockData.corp_code == corp_code).first()
    if stock_data is None:
        # Return a default object with all attributes set to 0
        return StockData(
            market_capitalization=0,
            per_value=0,
            pbr_value=0,
            cagr_1y=0,
            cagr_3y=0,
            cagr_5y=0,
            vol_1y=0,
            vol_3y=0,
            vol_5y=0
            # Add other fields if needed and set them to 0
        )
    return stock_data

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


async def get_stockgraph(stock_code: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = []
    max_pages = 9
    page = 1

    while page <= max_pages:
        url = f"https://finance.naver.com/item/sise_day.nhn?code={stock_code}&page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        
        soup = BeautifulSoup(response.text, 'html.parser')

        rows = soup.select("table.type2 tr")
        if not rows or len(rows) == 0:
            break  # 종료 조건: 더 이상 데이터가 없으면 종료

        page_data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                date_text = cols[0].get_text(strip=True)
                if date_text:
                    try:
                        date = datetime.strptime(date_text, '%Y.%m.%d').strftime('%Y-%m-%d')
                        if datetime.strptime(date, '%Y-%m-%d') >= start_date:
                            open_price = float(cols[3].get_text(strip=True).replace(',', ''))
                            high_price = float(cols[4].get_text(strip=True).replace(',', ''))
                            low_price = float(cols[5].get_text(strip=True).replace(',', ''))
                            close_price = float(cols[1].get_text(strip=True).replace(',', ''))
                            page_data.append({
                                "t": date,
                                "o": open_price,
                                "h": high_price,
                                "l": low_price,
                                "c": close_price
                            })
                    except ValueError:
                        continue

        stock_data.extend(page_data)

        # 페이지 증가
        page += 1

    stock_data.reverse()  # 데이터를 오래된 순으로 정렬
    return {"stock_data": stock_data[:90]}  # 최대 90일치 데이터만 반환




async def get_stockgraph1(stock_code: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    # 주식 코드에 올바른 접미사 추가
    stock_code = stock_code if '.' in stock_code else f'{stock_code}.KS'  # 예: 한국 주식의 경우 ".KS" 또는 ".KQ"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # yfinance를 사용하여 데이터 가져오기
    ticker = yf.Ticker(stock_code)
    hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # 데이터가 없는 경우 처리
    if hist.empty:
        raise ValueError(f"No data found for stock code: {stock_code}")

    stock_data = []
    for date, row in hist.iterrows():
        stock_data.append({
            "t": date.strftime('%Y-%m-%d'),
            "o": row['Open'],
            "h": row['High'],
            "l": row['Low'],
            "c": row['Close']
        })

    return {"stock_data": stock_data}