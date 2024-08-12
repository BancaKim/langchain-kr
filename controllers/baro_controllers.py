import base64
import os
from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct, text, func
from schemas.baro_schemas import CompanyInfoSchema

from services_def.baro_service import get_autocomplete_suggestions, get_corp_info_code, get_corp_info_jurir_no, get_corp_info_name, get_company_info, get_stockgraph, get_stockgraph1
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data, get_company_info_list, search_company, get_company_infoFS_list
from services_def.baro_service import get_custom_key_mapping, get_credit_ratings, get_top3_ratings
from services_def.baro_service import  FavoriteService, get_company_info, get_favorite_companies,  get_stockgraph1, generate_pdf, get_username_from_session
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
    username = request.session.get("username")
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
                "top3_rate": top3_rate,
                "username": username
            }
        )
    except Exception as e:
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


@baro.get("/baro_companyInfo3", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    username = request.session.get("username")
    
    company_info = get_company_info(db, jurir_no)
    FS2023 = get_FS2023(db, jurir_no)
    logger.info(f"FS2023: {FS2023}")
    # FS2022 = get_FS2022(db, jurir_no)
    # FS2021 = get_FS2021(db, jurir_no)
    # FS2020 = get_FS2020(db, jurir_no)
    # stock_data = get_Stock_data(db, company_info.corp_code)
    # stockgraph = await get_stockgraph1(company_info.stock_code)  # await 사용
    
    # adres = company_info.adres
    # kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    
    # custom_key_mapping = get_custom_key_mapping()
    # ratings = get_credit_ratings(db, jurir_no, custom_key_mapping)
    # top3_rate = get_top3_ratings(ratings)

    return templates.TemplateResponse("baro_service/baro_Test.html", {
        "request": request,
        "username": username,
        "company_info": company_info,
        "fs2023": FS2023,
        # "fs2022": FS2022,
        # "fs2021": FS2021,
        # "fs2020": FS2020,
        # "stock_data": stock_data,
        # "stockgraph": stockgraph,
        # "kakao_map_api_key": kakao_map_api_key,
        # "adres": adres,
        # "top3_rate": top3_rate
    })
    
    
   

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
@baro.get("/generate-pdf", response_class=FileResponse)
async def generate_pdf_endpoint(jurir_no: str, request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    company_info = get_company_info(db, jurir_no)
    
    # 예를 들어, 이미지 경로와 함께 Base64 문자열을 가져옵니다.
    financialbarchart = get_base64_image("static/images/financialbarchart.png")
    stockchart = get_base64_image("static/images/stockchart.png")
        
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
        "username": username
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


import base64
import os
from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
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
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data,  search_company, get_company_infoFS_list
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

@baro.get("/baro_companyInfo", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    username = request.session.get("username")
    print(jurir_no)
    company_info = get_company_info(db, jurir_no)
    FS2023 = get_FS2023(db, jurir_no)
    FS2022 = get_FS2022(db, jurir_no)
    FS2021 = get_FS2021(db, jurir_no)
    FS2020 = get_FS2020(db, jurir_no)
    print(company_info.corp_code)
    
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
    
    

@baro.post("/baro_companyInfo2")
async def read_company_info(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Form(None),
    search_type: Optional[str] = Form(None)
):
    username = request.session.get("username")
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
                
            else:
                print("Company info is None")
        else:
            print("Jurir_no is None")

        if not company_info:
            raise HTTPException(status_code=404, detail="Company not found")

        add_recent_view(db, username, company_info.corp_code)

        # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
        posts = db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()
        
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



## ####################################################################################################################################

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
@baro.get("/generate-pdf", response_class=FileResponse)
async def generate_pdf_endpoint(jurir_no: str, request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    company_info = get_company_info(db, jurir_no)
    
    # 예를 들어, 이미지 경로와 함께 Base64 문자열을 가져옵니다.
    financialbarchart = get_base64_image("static/images/financialbarchart.png")
    stockchart = get_base64_image("static/images/stockchart.png")
        
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
    
    
