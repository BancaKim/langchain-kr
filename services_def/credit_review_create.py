from contextlib import contextmanager
from dotenv import load_dotenv
from langchain_teddynote import logging
import os
import pandas as pd
import requests
import zipfile
import io
from lxml import etree
from langchain_community.document_loaders import BSHTMLLoader
import tempfile
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sqlalchemy import Column, DateTime, Integer, String, Text
from database import Base, SessionLocal
from typing import List, Dict, Any

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Spoon")

# 환경 변수에서 DART_API_KEY를 가져옵니다
DART_API_KEY = os.getenv("DART_API_KEY")


# 모델 정의
class ReportContent(Base):
    __tablename__ = "report_content"

    report_num = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(24))
    corp_name = Column(String(32))
    report_nm = Column(String(100))
    rcept_no = Column(String(32))
    rcept_dt = Column(DateTime)
    report_content = Column(Text)


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
        "pblntf_ty": "A",
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

    return ReportContent(
        corp_code=corp_code,
        corp_name=latest_report["corp_name"],
        report_nm=latest_report["report_nm"],
        rcept_no=latest_report["rcept_no"],
        rcept_dt=latest_report["rcept_dt"],
        report_content="",
    )


def fetch_document(rcept_no):
    url = "https://opendart.fss.or.kr/api/document.xml"
    params = {"crtfc_key": DART_API_KEY, "rcept_no": rcept_no}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API 요청 실패: 상태 코드 {response.status_code}")
    return response.content


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


def extract_audit_report(zip_content, rcp_no):
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            print("ZIP 파일 내용:")
            for file_info in zf.infolist():
                print(file_info.filename)

            audit_fnames = [
                info.filename
                for info in zf.infolist()
                if rcp_no in info.filename and info.filename.endswith(".xml")
            ]
            if not audit_fnames:
                raise ValueError("감사보고서 파일을 찾을 수 없습니다.")
            xml_data = zf.read(audit_fnames[0])

            # XML 파싱
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            root = etree.fromstring(xml_data, parser)

            # 세 부분 추출
            part1 = extract_section(root, "D-0-2-0-0", "D-0-3-0-0")
            part2 = extract_section(root, "D-0-3-1-0", "D-0-3-2-0")
            part3 = extract_section(root, "D-0-3-2-0", "D-0-3-3-0")

            # 세 부분 합치기
            extracted_xml = part1 + part2 + part3

            return extracted_xml

    except zipfile.BadZipFile:
        raise ValueError("ZIP 파일이 손상되었거나 유효하지 않습니다.")
    except etree.XMLSyntaxError as e:
        raise ValueError(f"XML 파싱 실패: {str(e)}")
    except IndexError:
        raise ValueError("필요한 TITLE 요소를 찾을 수 없습니다.")


def parse_html_from_xml(xml_data):
    parser = etree.HTMLParser()
    root = etree.fromstring(f"<html><body>{xml_data}</body></html>", parser)
    return root


def load_html_with_langchain(html_string):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_string)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path, open_encoding="utf-8")
        documents = loader.load()
        return documents
    finally:
        os.unlink(temp_file_path)


# Map 프롬프트 설정
map_template = """다음은 문서의 일부입니다:
{docs}
이 부분에서 주요 주제와 재무 정보를 포함한 핵심 내용을 100단어 이내로 요약해주세요.
요약:"""

map_prompt = PromptTemplate.from_template(map_template)

# LLM 모델 설정 (gpt-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce 프롬프트 설정
reduce_template = """
당신은 은행에서 대출을 심사하는 역할입니다.
당신은 대출 심사에 대한 판단 전에 신용평가보고서를 작성하고 있습니다.
다음은 요약들의 집합입니다: {docs}
이것들을 가져다가 최종적으로 통합하여 1.기업체개요 2.산업분석 3.영업현황 및 수익구조 4.재무구조 및 현금흐름 5.신용등급 부여의견으로 구분해서 요약해주세요.
각 섹션에 관련 재무 수치를 포함시키고 다섯줄 이상 작성해 주세요.

예시 :
1. 기업체 개요 : 동사 부동산 임대업 등의 사업목적으로 2001.10.16. 설립된 2023년말 기준 총자산 42,615백만원, 자본총계 22,335백만원, 매출액 3,502백만원,
당기순이익 37백만원 규모의 외감 소기업임.
2. 산업분석 : 최근 전방 산업 경기침체로 공실률 확대 기조 지속되어 매매가력 하락 및 임대소득 하락이 동시에 일어나 부동산 임대업 업황에 부정적인 영향을 미칠 가능성이 높음
3. 영업현황 및 수익구조 : 동사 2023년도 기준 매출액 전년도 대비 증가하였는 바, 안정적인 임대수입 영위 중에 있어 향후에도 구준한 매출액 시현에 따른 영업이익 지속 가능시됨.
4. 재무구조 및 현금흐름 : 동사 2023년말 기준 차입금 다소 증가하는 등 재무안정성 지표 상 미흡한 수준을 나타내고 있으나, 최근 3년간 무난한 현금흐름 나타내고 있으며, 지속적인 순이익 시현의 내부 유보로 자기자본 규모 확대되고 있음
5. 신용등급 부여의견 : 동사 최근 3년간 순이익 지속에 다른 영업활동 상 현금창출 지속되고, 순이익 시현의 내부유보로 자기자본 규모 확대되어 재무구조 개선되고 있으며, 향후에도 안정적인 영업실적 유지에 따른 수익성 유지로 채무상환 능력 인정됨.

요약된 내용:
"""

reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce 체인 설정
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000,
)

# MapReduce 체인 설정
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

# 텍스트 분할기 설정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=4000, chunk_overlap=0
)


def summarize_report(corp_code):
    try:
        # 보고서 번호 가져오기
        new_report = get_report(corp_code)
        rcept_no = new_report.rcept_no
        # 문서 가져오기
        zip_content = fetch_document(rcept_no)
        print("API 응답 크기:", len(zip_content), "바이트")

        # XML 데이터 추출 및 특정 섹션 파싱
        extracted_content = extract_audit_report(zip_content, rcept_no)
        print("XML 섹션 추출 완료")

        # HTML 파싱
        root = parse_html_from_xml(extracted_content)
        print("HTML 파싱 완료")

        # HTML을 문자열로 변환
        html_string = etree.tostring(
            root, pretty_print=True, method="html", encoding="unicode"
        )

        # LangChain을 사용하여 HTML 로드
        docs = load_html_with_langchain(html_string)
        print(f"추출된 문서 수: {len(docs)}")

        # 문서 분할
        split_docs = text_splitter.split_documents(docs)

        # MapReduce 체인 실행
        summary = map_reduce_chain.run(split_docs)

        # DB 저장
        new_report.report_content = summary
        with get_db_session() as session:
            session.add(new_report)

        return summary
    except Exception as e:
        print(f"Error in summarize_report: {str(e)}")
        return None
