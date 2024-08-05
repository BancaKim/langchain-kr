import os
from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct, text, func
from schemas.baro_schemas import CompanyInfoSchema
from services_def.baro_service import get_autocomplete_suggestions, get_corp_info_code, get_corp_info_jurir_no, get_corp_info_name, get_company_info, get_stockgraph, get_stockgraph1
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data, get_company_info_list, search_company, get_company_infoFS_list
import logging
from typing import Dict, List, Optional
from models.baro_models import CompanyInfo


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
baro = APIRouter()
templates = Jinja2Templates(directory="templates")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# 바로 등급 검색 페이지 / 나의업체현황/ 최근조회업체 return
@baro.get("/baro_companyList", response_class=HTMLResponse)
async def read_companyList(request: Request, search_value: str = "", db: Session = Depends(get_db)):
    username = request.session.get("username")
    
    jurir_no = search_company(db, search_value) if search_value else []
    my_jurir_no = ["1101110017990", "1101110019219", "1345110004412"]
    recent_jurir_no = ["1101110032154", "1201110018368", "1101110162191"]
    
    search_company_list = get_company_infoFS_list(db, jurir_no) if jurir_no else []
    
    my_company_list = get_company_infoFS_list(db, my_jurir_no) if my_jurir_no else []
    
    recent_view_list = get_company_infoFS_list(db, recent_jurir_no) if recent_jurir_no else []
    
    return templates.TemplateResponse(
        "baro_service/baro_companyList2.html", 
        {
            "request": request,
            "username": username,
            "search_company_list": search_company_list,
            "my_company_list": my_company_list,
            "recent_view_list": recent_view_list
        }
    )

@baro.get("/baro_companyInfo", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    username = request.session.get("username")
    
    company_info = get_company_info(db, jurir_no)
    FS2023 = get_FS2023(db, jurir_no)
    FS2022 = get_FS2022(db, jurir_no)
    FS2021 = get_FS2021(db, jurir_no)
    FS2020 = get_FS2020(db, jurir_no)
    stock_data=get_Stock_data(db, company_info.corp_code)
    stockgraph = await get_stockgraph1( company_info.stock_code)  # await 사용
    
    adres = company_info.adres
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    
    custom_key_mapping = {
    'AAA_plus': 'AAA+',
    'AAA': 'AAA',
    'AAA_minus': 'AAA-',
    'AA_plus': 'AA+',
    'AA': 'AA',
    'AA_minus': 'AA-',
    'A_plus': 'A+',
    'A': 'A',
    'A_minus': 'A-',
    'BBB_plus': 'BBB+',
    'BBB': 'BBB',
    'BBB_minus': 'BBB-',
    'BB_plus': 'BB+',
    'BB': 'BB',
    'BB_minus': 'BB-',
    'B_plus': 'B+',
    'B': 'B',
    'B_minus': 'B-',
    'CCC_plus': 'CCC+',
    'CCC': 'CCC',
    'CCC_minus': 'CCC-',
    'C': 'C',
    'D': 'D'
    }
    
    query1 = text("""
    SELECT AAA_plus, AAA, AAA_minus, AA_plus, AA, AA_minus, A_plus, A, A_minus, 
        BBB_plus, BBB, BBB_minus, BB_plus, BB, BB_minus, B_plus, B, B_minus, 
        CCC_plus, CCC, CCC_minus, C, D
    FROM spoon.predict_ratings
    WHERE base_year = '2023' AND corporate_number = :corporate_number
    ORDER BY timestamp DESC
    LIMIT 1;
    """)

    credit_rate = db.execute(query1, {"corporate_number": jurir_no}).fetchone()
    
    if not credit_rate:
        default_ratings = {
            'AAA+': 0,
            'AAA': 0,
            'AAA-': 0,
            'AA+': 0,
            'AA': 0,
            'AA-': 0,
            'A+': 0,
            'A': 0,
            'A-': 0,
            'BBB+': 0,
            'BBB': 0,
            'BBB-': 0,
            'BB+': 0,
            'BB': 0,
            'BB-': 0,
            'B+': 0,
            'B': 0,
            'B-': 0,
            'CCC+': 0,
            'CCC': 0,
            'CCC-': 0,
            'C': 0,
            'D': 0
        }
        ratings = default_ratings
    else:
        ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}

    # Sort and select top 3 ratings
    top3_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
    
    # Ensure at least 3 entries in top3_rate with default values if less than 3 available
    top3_rate = [{"key": column, "value": value} for column, value in top3_ratings]
    while len(top3_rate) < 3:
        top3_rate.append({"key": "N/A", "value": 0})  # Use "N/A" or other default key

    return templates.TemplateResponse("baro_service/baro_companyInfo.html", {
        "request": request,
        "username": username,
        "company_info": company_info,
        "fs2023": FS2023,
        "fs2022": FS2022,
        "fs2021": FS2021,
        "fs2020": FS2020,
        "stock_data": stock_data,
        "stockgraph": stockgraph,
        "kakao_map_api_key": kakao_map_api_key,
        "adres": adres,
        "top3_rate": top3_rate
    })

@baro.post("/baro_companyInfo2")
async def read_company_info(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Form(None),
    search_type: Optional[str] = Form(None)
):
    try:
        query = db.query(CompanyInfo)
        
        jurir_no = None
        company_info = None
        
        if name:
            if search_type == "company_name":
                result = db.query(CompanyInfo.jurir_no).filter(func.trim(CompanyInfo.corp_name) == name).first()
                if result:
                    jurir_no = result[0]
            elif search_type == "company_code":
                result = db.query(CompanyInfo.jurir_no).filter(func.trim(CompanyInfo.corp_code) == name).first()
                if result:
                    jurir_no = result[0]

        print("jurir_no:", jurir_no)  # Debug print to check if jurir_no is fetched

        if jurir_no:
            company_info = get_company_info(db, jurir_no)
            print("company_info:", company_info)  # Debug print to check if company_info is fetched

            if company_info:
                print("company_info.corp_code:", company_info.corp_code)  # Debug print to check corp_code

                FS2023 = get_FS2023(db, jurir_no)
                FS2022 = get_FS2022(db, jurir_no)
                FS2021 = get_FS2021(db, jurir_no)
                FS2020 = get_FS2020(db, jurir_no)
                stock_data = get_Stock_data(db, company_info.corp_code)
                stockgraph = await get_stockgraph1( company_info.stock_code)  # await 사용
                
                adres = company_info.adres
                kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
                           
                custom_key_mapping = {
                    'AAA_plus': 'AAA+',
                    'AAA': 'AAA',
                    'AAA_minus': 'AAA-',
                    'AA_plus': 'AA+',
                    'AA': 'AA',
                    'AA_minus': 'AA-',
                    'A_plus': 'A+',
                    'A': 'A',
                    'A_minus': 'A-',
                    'BBB_plus': 'BBB+',
                    'BBB': 'BBB',
                    'BBB_minus': 'BBB-',
                    'BB_plus': 'BB+',
                    'BB': 'BB',
                    'BB_minus': 'BB-',
                    'B_plus': 'B+',
                    'B': 'B',
                    'B_minus': 'B-',
                    'CCC_plus': 'CCC+',
                    'CCC': 'CCC',
                    'CCC_minus': 'CCC-',
                    'C': 'C',
                    'D': 'D'
                }
                
                query1 = text("""
                        SELECT AAA_plus, AAA, AAA_minus, AA_plus, AA, AA_minus, A_plus, A, A_minus, 
                            BBB_plus, BBB, BBB_minus, BB_plus, BB, BB_minus, B_plus, B, B_minus, 
                            CCC_plus, CCC, CCC_minus, C, D
                        FROM spoon.predict_ratings
                        WHERE base_year = '2023' AND corporate_number = :corporate_number
                        ORDER BY timestamp DESC
                        LIMIT 1;
                """)

                credit_rate = db.execute(query1, {"corporate_number": jurir_no}).fetchone()
    
                if not credit_rate:
                    default_ratings = {
                        'AAA+': 0,
                        'AAA': 0,
                        'AAA-': 0,
                        'AA+': 0,
                        'AA': 0,
                        'AA-': 0,
                        'A+': 0,
                        'A': 0,
                        'A-': 0,
                        'BBB+': 0,
                        'BBB': 0,
                        'BBB-': 0,
                        'BB+': 0,
                        'BB': 0,
                        'BB-': 0,
                        'B+': 0,
                        'B': 0,
                        'B-': 0,
                        'CCC+': 0,
                        'CCC': 0,
                        'CCC-': 0,
                        'C': 0,
                        'D': 0
                    }
                    ratings = default_ratings
                else:
                    ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}

                ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}
                top3_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
                
                top3_rate = [{"key": column, "value": value} for column, value in top3_ratings]      
                while len(top3_rate) < 3:
                    top3_rate.append({"key": "N/A", "value": 0})  # Use "N/A" or other default key          
                
            else:
                print("Company info is None")
        else:
            print("Jurir_no is None")

        if not company_info:
            raise HTTPException(status_code=404, detail="Company not found")

        return templates.TemplateResponse(
            "baro_service/baro_companyInfo.html",
            {
                "request": request,
                "company_info": company_info,
                "fs2023": FS2023,
                "fs2022": FS2022,
                "fs2021": FS2021,
                "fs2020": FS2020,
                "stock_data": stock_data,
                "stockgraph": stockgraph,  # stockgraph 변수를 템플릿에 전달
                "kakao_map_api_key": kakao_map_api_key, 
                "adres": adres,
                "top3_rate": top3_rate
            }
        )
    except Exception as e:
        print("An error occurred:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 법인명으로 섭외등록 연결 김철민 수정
@baro.get("/baro_contact", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), register: bool = False, db: Session = Depends(get_db)):
    company_info = get_company_info(db, jurir_no)
    
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    if register:
        # 법인명을 세션에 저장
        request.session['corporation_name'] = company_info.corp_name
        # 섭외등록 페이지로 리다이렉트
        return RedirectResponse(url="/contact/create")
    # logger.info(f"company_info.corp_code: {company_info.corp_code}")
    # return templates.TemplateResponse("baro_service/baro_companyInfo.html", {"request": request, "company_info": company_info})

# 지도에 주소 기반 검색 결과 표시 김철민 수정
@baro.get("/baro_map2", response_class=HTMLResponse)
async def get_map2(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    company_info = db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")

    adres = company_info.adres
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    return templates.TemplateResponse("contact/map2.html", {"request": request, "kakao_map_api_key": kakao_map_api_key, "adres": adres})

    
@baro.get("/test1234")
async def read_join(request: Request):
    return templates.TemplateResponse("baro_service/test.html", {"request": request})



@baro.get("/autocomplete", response_model=List[str])
async def autocomplete(
    query: str,
    search_type: str = Query("company_name", enum=["company_name", "company_code"]),
):
    db = SessionLocal()
    print(search_type)
    try:
        # Determine the column based on search_type
        if search_type == "company_name":
            column = "corp_name"
            print(column)
        elif search_type == "company_code":
            column = "corp_code"
            print(column)
        else:
            raise ValueError("Invalid search type")

        sql_query = text(
            f"""
            SELECT {column}
            FROM companyInfo
            WHERE {column} LIKE :query
            LIMIT 10
        """
        )
        results = db.execute(sql_query, {"query": f"%{query}%"}).fetchall()
        return [row[0] for row in results]  # Return list of results
    finally:
        db.close()



