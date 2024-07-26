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
from models.global_models import GlobalMarketing
from services_def.credit_companyinfo import get_autocomplete_suggestions
from services_def.credit_review_create import summarize_report
from dotenv import load_dotenv
from services_def.dependencies import get_db
from services_def.global_marketing_create import generate_report_html, get_report
from services_def.global_marketing_create import generate_and_save_report_html

marketing = APIRouter(prefix="/marketing")
templates = Jinja2Templates(directory="templates")


@marketing.get("/createMarketing/")
async def create_marketing(db: Session = Depends(get_db)):
    # finanical_summary_v2의 회사코드 리스트 가져오기
    # corp_code = "00164779"  # 에스케이하이닉스(주)
    # corp_code = "00126380"  # 삼성전자
    # corp_code = "00102618"  # 계양전기

    corp_codes = []
    corp_codes.append("00131832")  # 에스케이 디스커버리
    # corp_codes.append("00149655")  # 삼성물산

    # corp_codes.append("00155355")  # 풀무원
    # corp_codes.append("00105961")  # LG이노텍
    # corp_codes.append("00231707")  # 비트컴퓨터
    # corp_codes.append("00545929")  # 제넥신
    # corp_codes.append("01133217")  # 카카오뱅크
    # corp_codes.append("00117212")  # 두산
    # corp_codes.append("01105153")  # 두산로보틱스
    # corp_codes.append("00164742")  # 현대자동차
    # corp_codes.append("00164788")  # 현대모비스

    # 받아온 회사 코드로 최신 정기 공시보고서 번호 dart api로부터 받아오기
    results = []
    for corp_code in corp_codes:
        try:
            html_content = await generate_and_save_report_html(corp_code, db)
            results.append({"corp_code": corp_code, "status": "success"})
        except Exception as e:
            results.append(
                {"corp_code": corp_code, "status": "failed", "error": str(e)}
            )

    return {"result": "DB operation completed", "details": results}


@marketing.get("/api/companies")
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


@marketing.get("/api/companies/search")
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
    query = db.query(GlobalMarketing)

    if name:
        if search_type == "company_name":
            query = query.filter(GlobalMarketing.corp_name.ilike(f"%{name}%"))
        elif search_type == "company_code":
            query = query.filter(GlobalMarketing.corp_code.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    print(total_pages)
    # Apply pagination
    globalMarketings = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "globalMarketings": [
            content.to_dict() for content in globalMarketings
        ],  # Ensure you have a method to convert SQLAlchemy objects to dict
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }


@marketing.get("/globalList/")
async def global_list(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
):
    query = db.query(GlobalMarketing)
    if name:
        query = query.filter(GlobalMarketing.corp_name.ilike(f"%{name}%"))

    total = query.count()
    total_pages = ceil(total / per_page)
    print(total_pages)
    globalLists = query.offset((page - 1) * per_page).limit(per_page).all()

    return templates.TemplateResponse(
        "marketing/global_search.html",
        {
            "request": request,
            "globalLists": globalLists,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "search_term": name,
        },
    )


@marketing.get("/globalDetail/{corp_code}")
async def read_global(request: Request, corp_code: str, db: Session = Depends(get_db)):
    try:
        report = get_report(corp_code)
        rcept_no = report.rcept_no

        # 데이터베이스에서 HTML 컨텐츠 조회
        content = (
            db.query(GlobalMarketing)
            .filter(GlobalMarketing.rcept_no == rcept_no)
            .first()
        )

        if content:
            table = content.html_content
        else:
            # 데이터베이스에 없으면 생성 및 저장
            table = await generate_and_save_report_html(corp_code, db)

        return templates.TemplateResponse(
            "marketing/global_detail.html",
            {"request": request, "table": table},
        )
    except Exception as e:
        print(f"Error in read_global: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
