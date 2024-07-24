import asyncio
from math import ceil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
import urllib3
from models.baro_models import CompanyInfo
from models.credit_models import ReportContent
from schemas.credit_schemas import ReportContentSchema
from services_def.credit_companyinfo import get_corp_info_name, get_corp_info_code, get_autocomplete_suggestions
from services_def.credit_review_create import summarize_report
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from services_def.dependencies import get_db

credit = APIRouter(prefix="/credit")
templates = Jinja2Templates(directory="templates")

# 가상의 회사 데이터
companies = [
    {"name": "A 회사", "address": "서울시 강남구", "phone": "02-1234-5678"},
    {"name": "B 기업", "address": "경기도 성남시", "phone": "031-9876-5432"},
    {"name": "C 주식회사", "address": "인천시 부평구", "phone": "032-1111-2222"},
]


# .env 파일에서 환경 변수를 로드합니다
load_dotenv()


@credit.get("/createReview/")
async def create_review(db: Session = Depends(get_db)):
    # finanical_summary_v1의 회사코드 리스트 가져오기
    # corp_code = "00164779"  # 에스케이하이닉스(주)
    # corp_code = "00126380"  # 삼성전자
    corp_code = "00102618"  # 계양전기
    # 받아온 회사 코드로 최신 정기 공시보고서 번호 dart api로부터 받아오기
    summary = summarize_report(corp_code)
    return {"result": "DB success"}


@credit.get("/api/companies")
async def get_companies(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    total = db.query(CompanyInfo).count()
    companies = (
        db.query(CompanyInfo).offset((page - 1) * per_page).limit(per_page).all()
    )

    return {
        "companies": companies,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": ceil(total / per_page),
    }


@credit.get("/api/companies/search")
async def search_companies(
    db: Session = Depends(get_db),
    name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    query = db.query(ReportContent)
    if name:
        query = query.filter(ReportContent.corp_name.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    reportContents = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "reportContents": reportContents,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }


@credit.get("/creditReview/")
async def read_credit(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    query = db.query(ReportContent)
    if name:
        query = query.filter(ReportContent.corp_name.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    reportContents = query.offset((page - 1) * per_page).limit(per_page).all()

    return templates.TemplateResponse(
        "creditreview/review_search.html",
        {
            "request": request,
            "reportContents": reportContents,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "search_term": name,
        },
    )


@credit.get("/reviewDetail/{rcept_no}")
async def readcredit(request: Request, rcept_no: str, db: Session = Depends(get_db)):
    reportContent = (
        db.query(ReportContent).filter(ReportContent.rcept_no == rcept_no).first()
    )
    return templates.TemplateResponse(
        "creditreview/review_detail.html",
        {
            "request": request,
            "reportContent": reportContent,
        },
    )


@credit.get("/stream-content/{rcept_no}")
async def stream_content(rcept_no: str, db: Session = Depends(get_db)):
    reportContent = (
        db.query(ReportContent).filter(ReportContent.rcept_no == rcept_no).first()
    )
    content = reportContent.report_content if reportContent else ""

    async def content_generator():
        for char in content:
            if char == "\n":
                yield "<br>"  # HTML 줄바꿈 태그
            else:
                yield char
            await asyncio.sleep(0.005)  # 50ms 딜레이

    return StreamingResponse(content_generator(), media_type="text/html")


@credit.get("/credit_companyinfo")
async def search_corp(search_type: str, search_value: str, request: Request):
    
    if search_type == "corp_code":
        selectedcompany = get_corp_info_code(search_value)
    elif search_type == "corp_name":
        selectedcompany = get_corp_info_name(search_value)
    else:
        return JSONResponse(content={"error": "Invalid search type"}, status_code=400)

    print(type(selectedcompany))
    print(selectedcompany.corp_code)
    
    if selectedcompany:
        company_info = ReportContentSchema.from_orm(selectedcompany)
        print("A:" + company_info.json())
        return JSONResponse(content=company_info.dict())


@credit.get("/autocomplete")
async def autocomplete(search_type: str, query: str):
    suggestions = get_autocomplete_suggestions(search_type, query)
    return JSONResponse(content=suggestions)