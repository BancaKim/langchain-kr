import asyncio
from math import ceil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from database import SessionLocal
from models.baro_models import CompanyInfo
from models.credit_models import ReportContent
from services_def.credit_companyinfo import get_autocomplete_suggestions
from services_def.credit_review_create import summarize_report
from dotenv import load_dotenv
from services_def.credit_review_create2 import run_credit_evaluation
from services_def.dependencies import get_db

credit = APIRouter(prefix="/credit")
templates = Jinja2Templates(directory="templates")

# .env 파일에서 환경 변수를 로드합니다
load_dotenv()


# @credit.get("/createReview/")
# async def create_review(db: Session = Depends(get_db)):
#     # finanical_summary_v1의 회사코드 리스트 가져오기
#     # corp_code = "00164779"  # 에스케이하이닉스(주)
#     # corp_code = "00126380"  # 삼성전자
#     # corp_code = "00102618"  # 계양전기

#     corp_codes = []

#     corp_codes.append("00155355")  # 풀무원
#     # corp_codes.append("00105961")  # LG이노텍
#     # corp_codes.append("00231707")  # 비트컴퓨터
#     # corp_codes.append("00545929")  # 제넥신
#     # corp_codes.append("01133217")  # 카카오뱅크
#     # corp_codes.append("00117212")  # 두산
#     # corp_codes.append("01105153")  # 두산로보틱스
#     # corp_codes.append("00164742")  # 현대자동차
#     # corp_codes.append("00164788")  # 현대모비스

#     # 받아온 회사 코드로 최신 정기 공시보고서 번호 dart api로부터 받아오기
#     for corp_code in corp_codes:
#         summary = run_credit_evaluation(corp_code)
#     return {"result": "DB success"}


@credit.get("/api/companies")
async def get_companies(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
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
    search_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
):
    # Determine the column to filter on based on search_type
    if search_type == "company_name":
        column = "corp_name"
    elif search_type == "company_code":
        column = "corp_code"
    else:
        raise ValueError("Invalid search type")

    # Build the query
    query = db.query(ReportContent)

    if name:
        if search_type == "company_name":
            query = query.filter(ReportContent.corp_name.ilike(f"%{name}%"))
        elif search_type == "company_code":
            query = query.filter(ReportContent.corp_code.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    print(total_pages)
    # Apply pagination
    reportContents = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "reportContents": [
            content.to_dict() for content in reportContents
        ],  # Ensure you have a method to convert SQLAlchemy objects to dict
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
    per_page: int = Query(10, ge=1, le=100),
):
    query = db.query(ReportContent)
    if name:
        query = query.filter(ReportContent.corp_name.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    print(total_pages)
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


@credit.get("/autocomplete", response_model=List[str])
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
            FROM report_content
            WHERE {column} LIKE :query
            LIMIT 5
        """
        )
        results = db.execute(sql_query, {"query": f"{query}%"}).fetchall()
        return [row[0] for row in results]  # Return list of results
    finally:
        db.close()
