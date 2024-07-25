from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct, text, func
from schemas.baro_schemas import CompanyInfoSchema
from services_def.baro_service import get_autocomplete_suggestions, get_corp_info_code, get_corp_info_jurir_no, get_corp_info_name, get_company_info
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data, get_company_info_list, search_company, get_company_infoFS_list
import logging
from typing import List, Optional
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
            "search_company_list": search_company_list,
            "my_company_list": my_company_list,
            "recent_view_list": recent_view_list
        }
    )

@baro.get("/baro_companyInfo", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    company_info = get_company_info(db, jurir_no)
    FS2023 = get_FS2023(db, jurir_no)
    FS2022 = get_FS2022(db, jurir_no)
    FS2021 = get_FS2021(db, jurir_no)
    FS2020 = get_FS2020(db, jurir_no)
    stock_data=get_Stock_data(db, company_info.corp_code)
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    # logger.info(f"company_info.corp_code: {company_info.corp_code}")
    return templates.TemplateResponse("baro_service/baro_companyInfo.html", {"request": request, "company_info": company_info, "fs2023": FS2023, "fs2022": FS2022, "fs2021": FS2021, "fs2020": FS2020, "stock_data" : stock_data})

@baro.get("/baro_companyInfo2")
async def read_company_info(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Query(None),
    search_type: Optional[str] = Query(None)
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
                "stock_data": stock_data
            }
        )
    except Exception as e:
        print("An error occurred:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")


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
        results = db.execute(sql_query, {"query": f"{query}%"}).fetchall()
        return [row[0] for row in results]  # Return list of results
    finally:
        db.close()
