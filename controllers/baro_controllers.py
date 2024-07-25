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
from services_def.baro_service import get_FS2023, get_FS2022



baro = APIRouter()
templates = Jinja2Templates(directory="templates")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@baro.get("/test", response_class=HTMLResponse)
async def read_company_info(request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)):
    company_info = get_company_info(db, jurir_no)
    FS2023 = get_FS2023(db, jurir_no)
    FS2022 = get_FS2022(db, jurir_no)
        
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")
    return templates.TemplateResponse("baro_service/baro_template.html", {"request": request, "company_info": company_info, "fs2023": FS2023, "fs2022": FS2022})





# 바로 등급 검색 페이지
@baro.get("/baro")
async def read_join(request: Request):
    return templates.TemplateResponse("baro_service/baro_search.html", {"request": request})



@baro.get("/template")
async def read_join(request: Request):
    return templates.TemplateResponse("/template.html", {"request": request})



@baro.get("/test1234")
async def read_join(request: Request):
    return templates.TemplateResponse("baro_service/test.html", {"request": request})
