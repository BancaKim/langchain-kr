import base64
import os
from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct, text, func
from models.common_models import Post
from schemas.baro_schemas import CompanyInfoSchema
from services_def.baro_service import  FavoriteService, add_recent_view, get_company_info, get_favorite_companies, get_recent_views,  get_stockgraph1, generate_pdf, get_username_from_session
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data,  search_company, get_company_infoFS_list, get_sample_jurir_no
from services_def.baro_service import fs_db_insert, FS_update_DB_base, generate_credit
import time
import logging
from typing import Dict, List, Optional
from models.baro_models import CompanyInfo
import requests
from services_def.news import fetch_naver_news


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

# 데이터 삽입을 위한 임시 엔드 포인트


# 바로 등급 검색 페이지 / 나의업체현황/ 최근조회업체 return
@baro.get("/baro_companyList", response_class=HTMLResponse)
async def read_companyList(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    recent_views_companies = get_recent_views(db,username)

    return templates.TemplateResponse(
        "baro_service/baro_companyList2.html", 
        {
            "request": request,
            "username": username,
            "recent_views_companies" : recent_views_companies
        }
    )

# 하단 검색
@baro.get("/baro_companyInfo", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    username = request.session.get("username")
    
    company_info = get_company_info(db, jurir_no)
    corp_code = company_info.corp_code
    FS2023 = get_FS2023(db, jurir_no)


    if FS2023.totalAsset2023 == 0:
        print("FS2023.totalAsset2023 == 0")
        # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
        fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2023)
        fs_db_insert(db, jurir_no, fs_dict, 2023)
        FS2023 = get_FS2023(db, jurir_no)
    
    # FS2022 최근 데이터로 업데이트
    FS2022 = get_FS2022(db, jurir_no)
    
    if FS2022.totalAsset2022 == 0:  
        fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2022)
        fs_db_insert(db, jurir_no, fs_dict, 2022)
        FS2022 = get_FS2022(db, jurir_no)
            
    # FS2021 최근 데이터로 업데이트
    FS2021 = get_FS2021(db, jurir_no)
    
    if FS2021.totalAsset2021 == 0:
        # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
        fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2021)
        fs_db_insert(db, jurir_no, fs_dict, 2021)
        FS2021 = get_FS2021(db, jurir_no)
    
    # FS2020 최근 데이터로 업데이트
    FS2020 = get_FS2020(db, jurir_no)
    
    if FS2020.totalAsset2020 == 0:
        # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
        fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2020)
        fs_db_insert(db, jurir_no, fs_dict, 2020)
        FS2020 = get_FS2020(db, jurir_no)


    
    
    
    stock_data=get_Stock_data(db, company_info.corp_code)
    
    try:
        stockgraph = await get_stockgraph1(company_info.stock_code)  # await 사용
    except ValueError as e:
        print(f"Failed to retrieve stock graph: {e}")
        stockgraph = {"stock_data": []}  # 빈 데이터를 반환하거나 적절히 처리
    
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
                        top3_rate = [{"key": "credit_rate", "value": "None"}]
    else:
                        ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}
                        top3_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
                        top3_rate = [{"key": column, "value": value} for column, value in top3_ratings]
                        
                        # Ensure at least 3 entries in top3_rate with default values if less than 3 available
                        while len(top3_rate) < 3:
                            top3_rate.append({"key": "N/A", "value": 0})  # Use "N/A" or other default key

    add_recent_view(db, username, company_info.corp_code)
    
    # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
    posts = db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()
    
    # print(company_info.corp_name)
    # print("FS2023.totalAsset2023", FS2023.totalAsset2023)
    # print("FS2022.totalAsset2022", FS2022.totalAsset2022)
    # print("FS2021.totalAsset2021", FS2021.totalAsset2021)
    # print("FS2020.totalAsset2020", FS2020.totalAsset2020)
    
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
        "top3_rate": top3_rate,
        "posts": posts
    })
    
    
# 상단 검색
@baro.post("/baro_companyInfo2")
async def read_company_info(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Form(None),
    search_type: Optional[str] = Form(None)
):
    # 세션에서 사용자 이름을 가져옴
    username = request.session.get("username")
 
    try:
        # CompanyInfo 테이블에 대한 기본 쿼리 생성
        query = db.query(CompanyInfo)
        
        jurir_no = None
        company_info = None
        print(name)
        print(search_type)
        
        # name이 존재할 경우, search_type에 따라 회사 이름이나 코드로 검색
        if name:
            if search_type == "company_name":
                # 회사 이름으로 검색하여 jurir_no를 가져옴
                result = db.query(CompanyInfo.jurir_no).filter(func.trim(CompanyInfo.corp_name) == name).first()
                if result:
                    jurir_no = result[0]
            elif search_type == "company_code":
                # 회사 코드로 검색하여 jurir_no를 가져옴
                result = db.query(CompanyInfo.jurir_no).filter(func.trim(CompanyInfo.jurir_no) == func.trim(name)).first()
                if result:
                    jurir_no = result[0]

        print("jurir_no:", jurir_no)  # jurir_no가 제대로 가져와졌는지 확인하기 위한 디버그 출력

        # jurir_no가 존재할 경우, 해당 회사 정보를 가져옴
        if jurir_no:
            company_info = get_company_info(db, jurir_no)
            corp_code = company_info.corp_code
            print("company_info:", company_info)  # company_info가 제대로 가져와졌는지 확인하기 위한 디버그 출력

            if company_info:
                print("company_info.corp_code:", company_info.corp_code)  # corp_code 확인용 디버그 출력
                
                # FS2023 최근 데이터로 업데이트
                FS2023 = get_FS2023(db, jurir_no)
                
                if FS2023.totalAsset2023 == 0: # null일 경우 0 으로 리턴
                    print("FS2023.totalAsset2023 == 0")
                    # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
                    fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2023)
                    fs_db_insert(db, jurir_no, fs_dict, 2023)
                    FS2023 = get_FS2023(db, jurir_no)
                
                # FS2022 최근 데이터로 업데이트
                FS2022 = get_FS2022(db, jurir_no)
                
                if FS2022.totalAsset2022 == 0:  
                    fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2022)
                    fs_db_insert(db, jurir_no, fs_dict, 2022)
                    FS2022 = get_FS2022(db, jurir_no)
                        
                # FS2021 최근 데이터로 업데이트
                FS2021 = get_FS2021(db, jurir_no)
                
                if FS2021.totalAsset2021 == 0:
                    # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
                    fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2021)
                    fs_db_insert(db, jurir_no, fs_dict, 2021)
                    FS2021 = get_FS2021(db, jurir_no)
                
                # FS2020 최근 데이터로 업데이트
                FS2020 = get_FS2020(db, jurir_no)
                
                if FS2020.totalAsset2020 == 0:
                    # DART API와 연동하여 최신 재무정보 업데이트 (테스트용)
                    fs_dict=FS_update_DB_base(db, corp_code, company_info.corp_name, 2020)
                    fs_db_insert(db, jurir_no, fs_dict, 2020)
                    FS2020 = get_FS2020(db, jurir_no)
            
                                    
                # 주식 관련 데이터를 가져옴
                stock_data = get_Stock_data(db, company_info.corp_code)

                # 주식 그래프 데이터를 가져옴 (비동기 호출)
                try:
                    stockgraph = await get_stockgraph1(company_info.stock_code)  # await 사용
                except ValueError as e:
                    print(f"Failed to retrieve stock graph: {e}")
                    stockgraph = {"stock_data": []}  # 빈 데이터를 반환하거나 적절히 처리
       
                # 회사 주소와 카카오 맵 API 키 가져오기
                adres = company_info.adres
                kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")

                # 신용 등급 매핑 테이블 정의
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
                
                # 가장 최신 신용 등급 정보를 가져오기 위한 SQL 쿼리
                query1 = text("""
                        SELECT AAA_plus, AAA, AAA_minus, AA_plus, AA, AA_minus, A_plus, A, A_minus, 
                            BBB_plus, BBB, BBB_minus, BB_plus, BB, BB_minus, B_plus, B, B_minus, 
                            CCC_plus, CCC, CCC_minus, C, D
                        FROM spoon.predict_ratings
                        WHERE base_year = '2023' AND corporate_number = :corporate_number
                        ORDER BY timestamp DESC
                        LIMIT 1;
                """)

                # 쿼리를 실행하여 신용 등급 데이터를 가져옴
                credit_rate = db.execute(query1, {"corporate_number": jurir_no}).fetchone()
                
                if not credit_rate:
                    generate_credit(db,jurir_no)
                    credit_rate = db.execute(query1, {"corporate_number": jurir_no}).fetchone()
    
                if not credit_rate:
                    top3_rate = [{"key": "credit_rate", "value": "None"}]
                else:
                    # 신용 등급 데이터에서 상위 3개 등급을 선택하여 딕셔너리로 구성
                    ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}
                    top3_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
                    top3_rate = [{"key": column, "value": value} for column, value in top3_ratings]
                    
                    # 최소 3개의 항목을 보장하기 위해 부족한 경우 기본 값을 추가
                    while len(top3_rate) < 3:
                        top3_rate.append({"key": "N/A", "value": 0})  # "N/A" 또는 기본 값을 사용
                
            else:
                print("Company info is None")  # company_info가 없는 경우 디버그 출력
        else:
            print("Jurir_no is None")  # jurir_no가 없는 경우 디버그 출력

        # 회사 정보를 찾지 못한 경우 404 에러 발생
        if not company_info:
            raise HTTPException(status_code=404, detail="Company not found")

        # 최근 본 회사 목록에 추가
        add_recent_view(db, username, company_info.corp_code)

        # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
        posts = db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()
        
        print(company_info.corp_name)
        print("jurir_no", jurir_no)
        print(FS2023.totalAsset2023)
        print(FS2022.totalAsset2022)
        print(FS2021.totalAsset2021)
        print(FS2020.totalAsset2020)
        
        # 템플릿에 필요한 데이터를 포함하여 HTML 응답을 반환
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
                "top3_rate": top3_rate,
                "username": username,
                "posts": posts
            }
        )
        
    except Exception as e:
        # 오류 발생 시 콘솔에 출력하고 500 에러 발생
        print("An error occurred:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")



# 법인명으로 섭외등록 연결

@baro.get("/baro_contact", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), register: bool = False, db: Session = Depends(get_db)):
    company_info = get_company_info(db, jurir_no)
    
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    if register:
        # 법인명을 세션에 저장
        request.session['corporation_name'] = company_info.corp_name
        # 섭외등록 페이지로 리다이렉트
        return RedirectResponse(url="/contact5")
    # logger.info(f"company_info.corp_code: {company_info.corp_code}")
    # return templates.TemplateResponse("baro_service/baro_companyInfo.html", {"request": request, "company_info": company_info})

# 섭외등록 수정 후 8월 2주차 수정
@baro.post("/baro_contact", response_class=HTMLResponse)
async def register_company(
    request: Request,
    jurir_no: str = Form(...),
    db: Session = Depends(get_db)
):
    company_info = get_company_info(db, jurir_no)
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # 법인명을 세션에 저장
    request.session['corporation_name'] = company_info.corp_name
    
    # 섭외등록 페이지로 리다이렉트
    return RedirectResponse(url="/contact5", status_code=303)

# 지도에 주소 기반 검색 결과 표시 김철민 수정
@baro.get("/baro_map2", response_class=HTMLResponse)
async def get_map2(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    company_info = db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")

    adres = company_info.adres
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    return templates.TemplateResponse("contact/map2.html", {"request": request, "kakao_map_api_key": kakao_map_api_key, "adres": adres})


@baro.get("/baro_news2", response_class=HTMLResponse)
async def search_news_by_jurir_no(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    # 데이터베이스에서 법인명을 조회
    company_info = db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    
    
    try:
        news_articles = fetch_naver_news(company_info.corp_name)
        # print(f"News articles fetched for {company_info.corp_name}: {news_articles}")  # 디버깅용 로그 추가
        return templates.TemplateResponse("contact/news2.html", {"request": request, "news": news_articles, "corporation_name": company_info.corp_name})
    except HTTPException as e:
        return templates.TemplateResponse("contact/news2.html", {"request": request, "error": str(e), "news": [], "corporation_name": company_info.corp_name})


    


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
            column = "jurir_no"
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



## ####################################################################################################################################

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
@baro.get("/generate-pdf", response_class=FileResponse)
async def generate_pdf_endpoint(jurir_no: str, request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    company_info = get_company_info(db, jurir_no)
    
    # 예를 들어, 이미지 경로와 함께 Base64 문자열을 가져옵니다.
    financialbarchart = get_base64_image("./static/images/financialbarchart.png")
    stockchart = get_base64_image("./static/images/stockchart.png")
        
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    
    FS2023 = get_FS2023(db, jurir_no)
    FS2022 = get_FS2022(db, jurir_no)
    FS2021 = get_FS2021(db, jurir_no)
    FS2020 = get_FS2020(db, jurir_no)
    stock_data = get_Stock_data(db, company_info.corp_code)
    
    try:
        stockgraph = await get_stockgraph1(company_info.stock_code)
    except ValueError as e:
        print(f"Failed to retrieve stock graph: {e}")
        stockgraph = {"stock_data": []}
    
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
        top3_rate = [{"key": "credit_rate", "value": "None"}]
    else:
        ratings = {custom_key_mapping.get(k, k): v for k, v in credit_rate._mapping.items() if v is not None}
        top3_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
        top3_rate = [{"key": column, "value": value} for column, value in top3_ratings]
        
        while len(top3_rate) < 3:
            top3_rate.append({"key": "N/A", "value": 0})
            
            
    # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
    posts = db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()

    # 템플릿을 문자열로 렌더링
    html_content = templates.get_template("baro_service/spoon_report.html").render({
        "request": None,
        "company_info": company_info,
        "fs2023": FS2023,
        "fs2022": FS2022,
        "fs2021": FS2021,
        "fs2020": FS2020,
        "stock_data": stock_data,
        "stockgraph": stockgraph,
        "kakao_map_api_key": kakao_map_api_key,
        "adres": adres,
        "top3_rate": top3_rate,
        "financialbarchart_base64": f"data:image/png;base64,{financialbarchart}",
        "stockchart_base64": f"data:image/png;base64,{stockchart}",
        "username": username,
        "posts": posts
    })
    
    pdf_path = generate_pdf(html_content)
    return FileResponse(pdf_path, media_type='application/pdf', filename="Spoon_Report.pdf")


class ImageData(BaseModel):
    image: str

@baro.post("/upload-image/{image_type}")
async def upload_image(image_type: str, image_data: ImageData):
    try:
        # 데이터 URL에서 "data:image/png;base64," 제거
        base64_image = image_data.image.split(",")[1]
        image_binary = base64.b64decode(base64_image)

        # 이미지 파일명 설정
        image_path = os.path.join("static/images/", f"{image_type}.png")

        # 디렉토리 존재 여부 확인 및 생성
        os.makedirs("static/images/", exist_ok=True)

        # 이미지 저장
        with open(image_path, "wb") as f:
            f.write(image_binary)
        
        return {"message": f"Image saved as {image_path}"}
    
    except Exception as e:
        # 오류 발생 시 상세 정보와 함께 HTTP 500 오류 반환
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")
    
    
    
@baro.post("/api/favorite/{corp_code}")
async def toggle_favorite(corp_code: str, request: Request, db: Session = Depends(get_db)):
    username = get_username_from_session(request)
    print("post:" + username)
    service = FavoriteService(db)
    result = service.toggle_favorite(username, corp_code)
    return result

@baro.get("/api/favorite/{corp_code}")
async def check_favorite(corp_code: str, request: Request, db: Session = Depends(get_db)):
    username = get_username_from_session(request)
    print("get:" + username)
    service = FavoriteService(db)
    is_favorited = service.is_favorite(username, corp_code)
    return {"is_favorited": is_favorited}


@baro.get("/api/favorites")
async def read_favorites(request: Request, db: Session = Depends(get_db)):
    username = get_username_from_session(request)
    try:
        companies = get_favorite_companies(db, username)
        return companies  # 빈 리스트를 반환해도 예외를 던지지 않습니다.
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    
