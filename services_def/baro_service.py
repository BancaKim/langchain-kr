import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Union
from bs4 import BeautifulSoup
from fastapi import HTTPException, Request, requests ,status
import httpx
from jinja2 import Template
from database import SessionLocal
from sqlalchemy import func, cast, Integer
from sqlalchemy.orm import Session
from sqlalchemy import text
from models.baro_models import CompanyInfo, FS2023, FS2022, FS2021, FS2020, Favorite, RecentView, StockData

import zipfile
import io
from lxml import etree
import pandas as pd
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain



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
    # Retrieve the data from the database
    fs_data = db.query(FS2023).filter(FS2023.jurir_no == jurir_no).first()

    if fs_data and fs_data.totalAsset2023 > 0:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2023(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2023=0,
            totalDebt2023=0,
            totalEquity2023=0,
            capital2023=0,
            revenue2023=0,
            operatingIncome2023=0,
            earningBeforeTax2023=0,
            netIncome2023=0,
            debtRatio2023=0.0,
            margin2023=0.0,
            turnover2023=0.0,
            leverage2023=0.0,
            created_at=None
        )

def get_FS2022(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2022).filter(FS2022.jurir_no == jurir_no).first()

    if fs_data and fs_data.totalAsset2022 > 0:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2022(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2022=0,
            totalDebt2022=0,
            totalEquity2022=0,
            capital2022=0,
            revenue2022=0,
            operatingIncome2022=0,
            earningBeforeTax2022=0,
            netIncome2022=0,
            debtRatio2022=0.0,
            margin2022=0.0,
            turnover2022=0.0,
            leverage2022=0.0,
            created_at=None
        )

def get_FS2021(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2021).filter(FS2021.jurir_no == jurir_no).first()

    if fs_data and fs_data.totalAsset2021 > 0:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2021(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2021=0,
            totalDebt2021=0,
            totalEquity2021=0,
            capital2021=0,
            revenue2021=0,
            operatingIncome2021=0,
            earningBeforeTax2021=0,
            netIncome2021=0,
            debtRatio2021=0.0,
            margin2021=0.0,
            turnover2021=0.0,
            leverage2021=0.0,
            created_at=None

        )

def get_FS2020(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2020).filter(FS2020.jurir_no == jurir_no).first()

    if fs_data and fs_data.totalAsset2020 > 0:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2020(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2020=0,
            totalDebt2020=0,
            totalEquity2020=0,
            capital2020=0,
            revenue2020=0,
            operatingIncome2020=0,
            earningBeforeTax2020=0,
            netIncome2020=0,
            debtRatio2020=0.0,
            margin2020=0.0,
            turnover2020=0.0,
            leverage2020=0.0,
            created_at=None
        )

"""
def get_FS2022(db: Session, jurir_no: str):
    return db.query(FS2022).filter(FS2022.jurir_no == jurir_no).first()

def get_FS2021(db: Session, jurir_no: str):
    return db.query(FS2021).filter(FS2021.jurir_no == jurir_no).first()

def get_FS2020(db: Session, jurir_no: str):
    return db.query(FS2020).filter(FS2020.jurir_no == jurir_no).first()
"""    

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



# get_stockgraph1 함수
async def get_stockgraph1(stock_code: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    # 주식 코드에 올바른 접미사 추가
    stock_code = stock_code if '.' in stock_code else f'{stock_code}.KS'  # 예: 한국 주식의 경우 ".KS" 또는 ".KQ"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    try:
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
    
    except Exception as e:
        # 기타 예외 처리
        print(f"오류가 발생했습니다: {e}")
        raise ValueError(f"An unexpected error occurred while retrieving data for stock code: {stock_code}") from e
    



import pdfkit
import logging


def generate_pdf(html_content):
    path_to_wkhtmltopdf = 'C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe'  # 경로를 자신의 시스템에 맞게 수정
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

    options = {
        'page-size': 'A4',
        'encoding': 'UTF-8',
        'no-outline': None,
        'no-stop-slow-scripts': None,
        'enable-local-file-access': None,
        'zoom': '0.9',  # 확대 비율을 약간 줄여 내용이 잘리지 않도록 조정
        'custom-header': [
            ('Accept-Encoding', 'gzip')
        ],
        'print-media-type': None,
        'margin-top': '5mm',
        'margin-right': '5mm',
        'margin-bottom': '5mm',
        'margin-left': '5mm',
        'disable-smart-shrinking': None,  # 스마트 축소 비활성화
    }

    pdf_path = "C:/Users/BIT/Desktop/Spoon_Report.pdf"

    try:
        pdfkit.from_string(html_content, pdf_path, options=options, configuration=config)
    except Exception as e:
        logging.error(f'PDF generation failed: {e}')
    return pdf_path


from sqlalchemy.exc import IntegrityError

class FavoriteService:
    def __init__(self, db: Session):
        self.db = db

    def toggle_favorite(self, username: str, corp_code: str) -> dict:
        # Check if the favorite already exists
        favorite = self.db.query(Favorite).filter_by(username=username, corp_code=corp_code).first()
        
        try:
            if favorite:
                # If exists, remove it
                self.db.delete(favorite)
                self.db.commit()
                return {"is_favorited": False}
            else:
                # If not exists, add it
                new_favorite = Favorite(username=username, corp_code=corp_code)
                self.db.add(new_favorite)
                self.db.commit()
                return {"is_favorited": True}
        except Exception as e:  # Generic exception handling
            # Rollback the session if an error occurs
            self.db.rollback()
            # Log the error (could also use a logging framework)
            print(f"Error: {e}")
            # Handle the specific error (optional)
            raise

    def is_favorite(self, username: str, corp_code: str) -> bool:
        return self.db.query(Favorite).filter_by(username=username, corp_code=corp_code).first() is not None

    def get_favorites_for_user(self, username: str):
        return self.db.query(Favorite).filter_by(username=username).all()
    
    
def get_username_from_session(request: Request):
    username = request.session.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )       
    return username



def get_favorite_companies(db: Session, username: str):
    favorites = db.query(Favorite).filter(Favorite.username == username).all()
    favorite_companies = []
    for favorite in favorites:
        company_info = db.query(CompanyInfo).filter(CompanyInfo.corp_code == favorite.corp_code).first()
        if company_info:
            financial_info = db.query(FS2023).filter(FS2023.jurir_no == company_info.jurir_no).first()
            favorite_companies.append({
                "corp_code": company_info.corp_code,
                "jurir_no": company_info.jurir_no,
                "corp_name": company_info.corp_name,
                "ceo_nm": company_info.ceo_nm,
                "corp_cls": company_info.corp_cls,
                "totalAsset2023": financial_info.totalAsset2023 // 100000000 if financial_info else None,
                "capital2023": financial_info.capital2023 // 100000000 if financial_info else None,
                "revenue2023": financial_info.revenue2023 // 100000000 if financial_info else None,
                "netIncome2023": financial_info.netIncome2023 // 100000000 if financial_info else None,

            })
    return favorite_companies




def add_recent_view(db: Session, username: str, corp_code: str):
    # Check if the recent view already exists
    existing_view = db.query(RecentView).filter(RecentView.username == username, RecentView.corp_code == corp_code).first()
    
    if existing_view:
        # Update the timestamp if the view exists
        existing_view.created_at = datetime.utcnow()
    else:
        # Add a new view if it doesn't exist
        new_view = RecentView(username=username, corp_code=corp_code, created_at=datetime.utcnow())
        db.add(new_view)

    # Remove the oldest view if more than 5 views are present
    recent_views = db.query(RecentView).filter(RecentView.username == username).order_by(RecentView.created_at.desc()).all()
    if len(recent_views) > 5:
        oldest_view = recent_views[-1]
        db.delete(oldest_view)
    
    db.commit()

    return existing_view if existing_view else new_view




def get_recent_views(db: Session, username: str):
    recent_views = db.query(RecentView).filter(RecentView.username == username).order_by(RecentView.created_at.desc()).limit(5).all()
    
    recent_views_companies = []
    for view in recent_views:
        company_info = db.query(CompanyInfo).filter(CompanyInfo.corp_code == view.corp_code).first()
        if company_info:
            financial_info = db.query(FS2023).filter(FS2023.jurir_no == company_info.jurir_no).first()
            recent_views_companies.append({
                "corp_code": company_info.corp_code,
                "jurir_no": company_info.jurir_no,
                "corp_name": company_info.corp_name,
                "corp_code": company_info.corp_code,
                "ceo_nm": company_info.ceo_nm,
                "corp_cls": company_info.corp_cls,
                "totalAsset2023": financial_info.totalAsset2023 // 100000000 if financial_info else None,
                "capital2023": financial_info.capital2023 // 100000000 if financial_info else None,
                "revenue2023": financial_info.revenue2023 // 100000000 if financial_info else None,
                "netIncome2023": financial_info.netIncome2023 // 100000000 if financial_info else None,
            })
    return recent_views_companies