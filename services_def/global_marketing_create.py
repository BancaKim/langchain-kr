import asyncio
from contextlib import contextmanager
import os
import aiohttp
import requests
import zipfile
import io
from typing import List, Dict, Any
from lxml import etree
from langchain_community.document_loaders import BSHTMLLoader
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
import logging
from fastapi import HTTPException
from dotenv import load_dotenv

from database import SessionLocal

# 환경 변수 로드
load_dotenv()

# API KEY 정보 로드
DART_API_KEY = os.getenv("DART_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 상수 정의
API_URL = "https://opendart.fss.or.kr/api/document.xml"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 0
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0

import asyncio
import os
import aiohttp
import requests
import zipfile
import io
from typing import List, Dict, Any
from lxml import etree
from langchain_community.document_loaders import BSHTMLLoader
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from models.global_models import GlobalMarketing


# 환경 변수 로드
load_dotenv()

# API KEY 정보 로드
DART_API_KEY = os.getenv("DART_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# 보고서 번호 불러오기
def get_report(corp_code):
    url_json = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": "20230101",
        "end_de": "20240630",
        "pblntf_detail_ty": "A001",
    }

    response = requests.get(url_json, params=params)
    data = response.json()
    data_list = data.get("list")
    df_list = pd.DataFrame(data_list)
    if df_list.empty:
        raise ValueError(f"No data found for corporation code: {corp_code}")

    # rcept_dt를 datetime 형식으로 변환 및 최신건 추출
    df_list["rcept_dt"] = pd.to_datetime(df_list["rcept_dt"])
    latest_report = df_list.sort_values("rcept_dt", ascending=False).iloc[0]

    return GlobalMarketing(
        corp_code=corp_code,
        corp_name=latest_report["corp_name"],
        report_nm=latest_report["report_nm"],
        rcept_no=latest_report["rcept_no"],
        rcept_dt=latest_report["rcept_dt"],
        html_content="",
    )


async def fetch_document(rcept_no: str) -> bytes:
    params = {"crtfc_key": DART_API_KEY, "rcept_no": rcept_no}
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params=params) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status, detail="DART API 요청 실패"
                )
            return await response.read()


def extract_section(root, start_aassocnote, end_aassocnote):
    start_element = root.xpath(
        f"//TITLE[@ATOC='Y' and @AASSOCNOTE='{start_aassocnote}']"
    )[0]
    end_element = root.xpath(f"//TITLE[@ATOC='Y' and @AASSOCNOTE='{end_aassocnote}']")[
        0
    ]

    extracted_elements = []
    current_element = start_element
    while current_element is not None:
        extracted_elements.append(
            etree.tostring(current_element, encoding="unicode", with_tail=True)
        )
        if current_element == end_element:
            break
        current_element = current_element.getnext()

    return "".join(extracted_elements)


async def extract_audit_report(zip_content: bytes, rcept_no: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            audit_fnames = [
                info.filename
                for info in zf.infolist()
                if rcept_no in info.filename and info.filename.endswith(".xml")
            ]
            if not audit_fnames:
                raise ValueError("감사보고서 파일을 찾을 수 없습니다.")

            xml_data = zf.read(audit_fnames[0])
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            root = etree.fromstring(xml_data, parser)

            return extract_section(root, "D-0-11-2-0", "D-0-11-3-0")
    except Exception as e:
        print(f"감사보고서 추출 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="감사보고서 추출 중 오류 발생")


def parse_html_from_xml(xml_data: str) -> etree.Element:
    parser = etree.HTMLParser()
    return etree.fromstring(f"<html><body>{xml_data}</body></html>", parser)


async def load_html_with_langchain(html_string: str) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_string)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path, open_encoding="utf-8")
        documents = await asyncio.to_thread(loader.load)
        return documents
    finally:
        os.unlink(temp_file_path)


def tsv_to_html_table(tsv_string: str) -> str:
    # 줄바꿈으로 행을 분리
    rows = tsv_string.strip().split("\n")
    # 각 행을 탭으로 분리하여 열 생성
    table_data = [row.split("\t") for row in rows]

    # HTML 테이블 생성
    html_table = '<table class="table table-striped table-bordered">'
    for row in table_data:
        html_table += "<tr>"
        for cell in row:
            html_table += f"<td>{cell}</td>"
        html_table += "</tr>"
    html_table += "</table>"

    return html_table


async def generate_report_html(rcept_no: str) -> str:
    try:
        zip_content = await fetch_document(rcept_no)
        print(f"API 응답 크기: {len(zip_content)} 바이트")

        extracted_content = await extract_audit_report(zip_content, rcept_no)
        print("XML 섹션 추출 완료")

        root = parse_html_from_xml(extracted_content)
        html_string = etree.tostring(
            root, pretty_print=True, method="html", encoding="unicode"
        )

        docs = await load_html_with_langchain(html_string)
        print(f"추출된 문서 수: {len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        template = """이 파일에서 '해외채무보증' 또는 '채무보증내역' 제목 아래에 있는 표를 읽어 TSV(Tab-Separated Values) 형식으로 변환해주세요. 다음 지침을 따라주세요:

        1. 표의 두 겹 칼럼 구조를 단일 행 헤더로 변환하세요. 상위 칼럼과 하위 칼럼을 언더스코어(_)로 결합하여 새로운 칼럼 이름을 만드세요.
        예: '채무보증금액'의 하위 칼럼 '제59기말'은 '채무보증금액_제59기말'로 변환

        2. 결과 TSV의 헤더는 다음과 같은 형식이어야 합니다 (실제 칼럼 이름은 원본 표에 따라 다를 수 있음):
        성명    관계    채권자    보증건수    보증기간_시작일    보증기간_종료일    채무보증금액_제59기말    채무보증금액_증가    채무보증금액_감소    채무보증금액_제60기말    채무금액

        3. 데이터 행은 각 칼럼에 해당하는 값을 포함해야 합니다. 값이 없는 경우 빈 칸으로 두지 말고 '-'로 표시하세요.

        4. 숫자 데이터는 쉼표나 기타 구분자 없이 순수한 숫자로 표현하세요.

        5. 날짜는 'YYYY-MM-DD' 형식으로 통일하세요.

        6. TSV 데이터만 반환하세요. 추가 설명이나 주석은 포함하지 마세요.

        7. 결과에 따옴표(''')나 기타 구분자를 포함하지 말고, 순수한 TSV 데이터만 반환하세요.

        8. 각 열은 반드시 탭(\t) 문자로 구분되어야 합니다. 열 사이에 공백이나 다른 문자를 사용하지 마세요.

        #Context:
        {context}

        #Answer:
        """
        prompt = PromptTemplate.from_template(template)
        llm = ChatOpenAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        context = retriever.get_relevant_documents("")
        tsv_result = await chain.arun(context=context)

        logging.info(f"Raw TSV result:\n{tsv_result}")

        # TSV 데이터 정제
        cleaned_tsv = "\n".join(
            line.strip() for line in tsv_result.split("\n") if line.strip()
        )

        html_table = tsv_to_html_table(cleaned_tsv)
        return html_table

    except Exception as e:
        print(f"보고서 요약 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"보고서 생성 중 오류 발생: {str(e)}"
        )


async def generate_and_save_report_html(corp_code: str, db: Session) -> str:
    try:
        report = get_report(corp_code)
        rcept_no = report.rcept_no

        # 기존 HTML 컨텐츠 확인
        existing_content = (
            db.query(GlobalMarketing)
            .filter(GlobalMarketing.rcept_no == rcept_no)
            .first()
        )
        if existing_content:
            return existing_content.html_content

        # 새 HTML 생성
        html_table = await generate_report_html(rcept_no)

        # 데이터베이스에 저장
        new_content = GlobalMarketing(
            corp_code=report.corp_code,
            corp_name=report.corp_name,
            report_nm=report.report_nm,
            rcept_no=rcept_no,
            rcept_dt=report.rcept_dt,
            html_content=html_table,
        )
        db.add(new_content)
        db.commit()
        db.refresh(new_content)

        return html_table

    except Exception as e:
        print(f"보고서 생성 및 저장 중 오류 발생: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"보고서 생성 및 저장 중 오류 발생: {str(e)}"
        )
