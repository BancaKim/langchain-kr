import asyncio
from math import ceil
from typing import List, Optional, TypedDict
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from database import SessionLocal
from models.baro_models import CompanyInfo
from models.credit_models import ReportContent
from dotenv import load_dotenv
from services_def.chatbot_logic import generate_response, setup_chatbot
from services_def.dependencies import get_db
from langchain_upstage import UpstageGroundednessCheck
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from services_def.rag.utils import format_docs, format_searched_docs
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults

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
    username = request.session.get("username")
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
            "username": username,
            "reportContents": reportContents,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "search_term": name,
        },
    )


@credit.get("/reviewDetail/{rcept_no}/{corp_code}")
async def readcredit(
    request: Request,
    rcept_no: str,
    corp_code: str = None,
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    try:
        # Fetch the report content based on rcept_no
        reportContent = (
            db.query(ReportContent).filter(ReportContent.rcept_no == rcept_no).first()
        )
        if not reportContent:
            raise HTTPException(status_code=404, detail="Report content not found")

        company_info = None
        if corp_code:
            # Fetch company info based on corp_code if provided
            company_info = (
                db.query(CompanyInfo).filter(CompanyInfo.corp_code == corp_code).first()
            )
            if company_info:
                print(
                    "company_info:", company_info
                )  # Debug print to check if company_info is fetched
            else:
                print("Company info not found for corp_code:", corp_code)

        return templates.TemplateResponse(
            "creditreview/review_detail.html",
            {
                "request": request,
                "username": username,
                "reportContent": reportContent,
                "company_info": company_info,
            },
        )
    except Exception as e:
        print("An error occurred:", str(e))  # Debug print
        raise HTTPException(status_code=500, detail="Internal Server Error")


@credit.get("/stream-content/{rcept_no}")
async def stream_content(rcept_no: str, db: Session = Depends(get_db)):
    reportContent = (
        db.query(ReportContent).filter(ReportContent.rcept_no == rcept_no).first()
    )
    content = reportContent.report_content if reportContent else ""

    # 문단 구분 (예: 특정 키워드를 사용하여 문단 나누기)
    sections = content.split("\n\n")
    paragraphs = {
        "company_overview": sections[0] if len(sections) > 0 else "",
        "industry_analysis": sections[1] if len(sections) > 1 else "",
        "operational_status": sections[2] if len(sections) > 2 else "",
        "financial_structure": sections[3] if len(sections) > 3 else "",
        "credit_rating_opinion": sections[4] if len(sections) > 4 else "",
    }

    # 제목 매핑
    titles = {
        "company_overview": "기업체 개요",
        "industry_analysis": "산업 분석",
        "operational_status": "영업 현황 및 수익 구조",
        "financial_structure": "재무 구조 및 현금 흐름",
        "credit_rating_opinion": "신용 등급 부여 의견",
    }

    def extract_content(text):
        delimiter = "\n"
        start_index = text.find(delimiter)
        if start_index != -1:
            return text[start_index + len(delimiter) :].strip()
        return text

    async def content_generator():
        for key in [
            "company_overview",
            "industry_analysis",
            "operational_status",
            "financial_structure",
            "credit_rating_opinion",
        ]:
            paragraph_content = extract_content(paragraphs[key])
            html_content = (
                "<div class='flex-1 bg-white rounded-lg shadow-lg'>"
                f"<div class='bg-gray-100 border-b border-gray-300 p-3 rounded-t-lg'>"
                f"<h2 class='text-xl font-semibold text-gray-800'>{titles[key]}</h2></div>"
                f"<div class='p-6 text-lg'>{paragraph_content}</div></div><br>"
            )

            # print(f"Generated HTML for {key}: {html_content}")
            # print(paragraphs[key])
            yield html_content
            await asyncio.sleep(0.1)  # 소량의 대기시간을 추가하여 스트리밍처럼 보이도록

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

@credit.get("/chatMessage")
async def getMessage(    
    request: Request,
    corp_code: str = None,
    user_input: str = None,
    db: Session = Depends(get_db)):
    
    reportContent = db.query(ReportContent).filter(ReportContent.corp_code == corp_code).first()
    content = reportContent.report_content
    chatbot_app = setup_chatbot(content)
    
    answer = generate_response(chatbot_app, user_input)
    
    return JSONResponse(content={"answer": answer})