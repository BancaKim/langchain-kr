from fastapi import APIRouter, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import date
from database import SessionLocal
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy import distinct
from schemas.baro_schemas import CompanyInfoSchema
from services_def.baro_service import get_autocomplete_suggestions, get_corp_info_code, get_corp_info_jurir_no, get_corp_info_name



baro = APIRouter()
templates = Jinja2Templates(directory="templates")

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

    print(type(selectedcompany))
    print(selectedcompany.corp_code)
    
    if selectedcompany:
        company_info = CompanyInfoSchema.from_orm(selectedcompany)
        print("A:" + company_info.json())
        return JSONResponse(content=company_info.dict())
    
    
    
@baro.get("/autocomplete")
async def autocomplete(search_type: str, query: str):
    suggestions = get_autocomplete_suggestions(search_type, query)
    return JSONResponse(content=suggestions)