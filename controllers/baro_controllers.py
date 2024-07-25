from fastapi import APIRouter, Form, HTTPException, Request, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct
from schemas.baro_schemas import CompanyInfoSchema
from services_def.baro_service import get_autocomplete_suggestions, get_corp_info_code, get_corp_info_jurir_no, get_corp_info_name, get_company_info
from services_def.baro_service import get_FS2023, get_FS2022, get_FS2021, get_FS2020, get_Stock_data, get_company_info_list, search_company
import logging
from typing import List

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
    search_company_list = get_company_info_list(db, jurir_no) if jurir_no else []
    my_company_list = get_company_info_list(db, my_jurir_no) if my_jurir_no else []
    recent_view_list = get_company_info_list(db, recent_jurir_no) if recent_jurir_no else []
    
    return templates.TemplateResponse(
        "baro_service/baro_companyList.html", 
        {
            "request": request,
            "search_company_list": search_company_list,
            "my_company_list": my_company_list,
            "recent_view_list": recent_view_list
        }
    )


# # 바로 등급 검색 페이지
# @baro.get("/baro_companyList")
# async def read_companyList(request: Request, jurir_no: List[str] = None, db: Session = Depends(get_db)):
    
#     jurir_no = ["1101110032154", "1201110018368", "1101110162191", "1101110017990", "1101110019219", "1345110004412"]
#     if jurir_no:
#         company_list = get_company_info_list(db, jurir_no)
#     else:
#         company_list = []
        
#     return templates.TemplateResponse("baro_service/baro_companyList.html", {"request": request, "company_list": company_list})




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



# 바로 등급 검색 페이지
@baro.get("/baro")
async def read_join(request: Request):
    return templates.TemplateResponse("baro_service/baro_search.html", {"request": request})



@baro.get("/baro_info")
async def search_corp(search_type: str, search_value: str, request: Request):
    
    if search_type == "corp_code":
        selectedcompany = get_corp_info_code(search_value)
    elif search_type == "corp_name":
        selectedcompany = get_corp_info_name(search_value)
    elif search_type == "jurir_no":
        selectedcompany = get_corp_info_jurir_no(search_value)
    else:
        return JSONResponse(content={"error": "Invalid search type"}, status_code=400)

    #print(type(selectedcompany))
    #print(selectedcompany.corp_code)
    
    if selectedcompany:
        company_info = CompanyInfoSchema.from_orm(selectedcompany)
        print("A:" + company_info.json())
        return JSONResponse(content=company_info.dict())
    
    
    
@baro.get("/autocomplete")
async def autocomplete(search_type: str, query: str):
    suggestions = get_autocomplete_suggestions(search_type, query)
    return JSONResponse(content=suggestions)




@baro.get("/template")
async def read_join(request: Request):
    return templates.TemplateResponse("/template.html", {"request": request})
