import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Union
import re
from bs4 import BeautifulSoup
from fastapi import HTTPException, Request, requests ,status
import requests as http_requests
import httpx
from jinja2 import Template
from database import SessionLocal, crtfc_key
from sqlalchemy import func, cast, Integer
from sqlalchemy.orm import Session, aliased
from sqlalchemy import text, Table, MetaData
from models.baro_models import CompanyInfo, FS2023, FS2022, FS2021, FS2020, Favorite, RecentView, StockData
import requests as http_requests
import zipfile
import io
from lxml import etree
import pandas as pd
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging as langchain_logging
import pdfkit
import logging
import json

import os
from dotenv import load_dotenv

# 환경 변수에서 DART_API_KEY를 가져옵니다
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

# API KEY 정보로드
load_dotenv()

# langchain_teddynote의 로깅 설정
langchain_logging.langsmith("Spoon")

# LOGGING 모듈을 사용하여 일반적인 로깅 설정
logging.basicConfig(level=logging.INFO)


# httpx 및 requests 라이브러리의 디버그 로그를 비활성화
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)  # 추가된 부분

def search_company(db: Session, keyword: str) -> List[str]:

    keyword_pattern = f"%{keyword}%"
    
    query = text("""
        SELECT jurir_no 
        FROM companyInfo 
        WHERE corp_name LIKE :keyword_pattern 
        OR jurir_no = :keyword 
        OR bizr_no = :keyword
    """)
    result = db.execute(query, {"keyword_pattern": keyword_pattern, "keyword": keyword})
    jurir_nos = [row.jurir_no for row in result.fetchall()]
    print(f"Found jurir_no: {jurir_nos}")  # 터미널에 출력
    return jurir_nos


def FS_update(db: Session, corp_code: str, corp_name: str, baseYear):
    # FS 업데이트 시작 로그
    print(f"FS update started for {corp_name} (corp_code: {corp_code}, year: {baseYear})")
    
    # Step 1: DART 공시 보고서 번호를 받아옵니다.
    report_list = get_dart_report_list(corp_code)
    # print(f"Report list fetched for {corp_name}: {report_list}")

    valid_receipt_no = None
    for i in range(2):  # 2번 반복
              
        # Step 2: 기준 연도의 최신 보고서 접수번호를 가져옵니다.
        FSSreport = get_latest_report_receipt_no(report_list, baseYear, valid_receipt_no)
        print(f"Latest report receipt number for {baseYear}: {FSSreport}")

        # Step 3: 보고서 번호로 DART 사이트에서 필요한 데이터를 추출합니다.
        soup, url = fetch_and_create_urls(FSSreport, 1)
        
        if soup: break
        else: valid_receipt_no = FSSreport
    
    
    print(f"Processing URL for {corp_name}: {url}")
    # print(f"Processing soup {soup}")
    # Step 4: 추출된 데이터에서 계정 정보와 당기 금액을 정리합니다.
    extracted_data = None
    if soup:
        # 첫 번째 시도 로그
        # print(f"First attempt to extract data for {corp_name}")
        extracted_data = extract_account_data(soup)
        # print(f"Extracted Data (1st try) for {corp_name}: {extracted_data}")

        if extracted_data == []:
            # print(f"First extraction attempt failed, trying again for {corp_name}")
            soup, url = fetch_and_create_urls(FSSreport, 2)
            extracted_data = extract_account_data(soup)
            # print(f"Extracted Data (2nd try) for {corp_name}: {extracted_data}")
        
    if not extracted_data:
        print(f"Failed to retrieve or parse the report for {corp_name}")
        # print(soup)
        # fs_dict 초기화
        fs_dict = {
            "자산총계": 0,
            "부채총계": 0,
            "자본총계": 0,
            "자본금": 0,
            "매출액": 0,
            "영업이익": 0,
            "법인세차감전순이익": 0,
            "당기순이익": 0,
            "재무제표url" : url,
            "재무제표상세" : None
        }
        print(f"Returning default fs_dict for {corp_name}")
        return fs_dict  # 기본 값을 반환하여 에러 방지
    
    # Step 5: 요약된 재무 데이터를 처리
    fs_dict = FSmaker(extracted_data, url)
    
    
    # Step 5: fs_dict에 None 값이 있는지 확인하고, 있을 경우 GPT로 업데이트
    # 예를 들어 국민은행 2020~2021 재무제표 양식이 완전히 달라 스크래핑 할 수 없음. 
    # if any(value is None for value in fs_dict.values()):
    #     print(f"Some values are missing in fs_dict for {corp_name}, updating with GPT")
    #     # HTML과 기존 재무 정보를 gpt에 넘긴다. 
    #     extracted_data = extract_account_data_with_gpt(soup.text)
    #     print(f"fs_dict after GPT update for {corp_name}: {extracted_data}")
        
        
    
    
    # print(f"Final financial statement dictionary for {corp_name}: {fs_dict}")
    print(f"Processing URL for {corp_name}: {url}")


    # FS 업데이트 완료 로그
    # print(f"FS update completed for {corp_name} (corp_code: {corp_code}, year: {baseYear})")
    return fs_dict


from langchain_openai import ChatOpenAI
import json

def extract_account_data_with_gpt(extracted_text: str) -> List[dict]:
    llm = ChatOpenAI(api_key=GPT_API_KEY, model_name="gpt-4o-mini-2024-07-18", temperature=0.0)

    # 프롬프트를 텍스트로 정의합니다.
    prompt = f"""
    아래는 기업의 재무제표와 관련된 텍스트 데이터입니다. 이 데이터를 바탕으로 계정 이름과 금액을 추출하여 다음 형식의 리스트로 반환하십시오:
    [
        {{"account_name": "<계정 이름>", "current_amount": <금액>}},
        ...
    ]
    금액은 숫자만 포함하고, 단위는 계산해서 원단위로 환산해주세요. 
    만약 값이 음수라면, '-' 기호를 포함하여 반환해주세요.
    
    자산총계, 부채총계, 자본총계, 자본금, 매출액, 영업이익, 법인세차감전순이익, 당기순이익은 꼭 포함해주세요.
    매출액이 없을 경우 이자수익과 수수료수익 또는 보험영업수익과 투자영업수익을 꼭 포함해주세요. 
    
    각각의 계정은 아래와 같이 매핑될 수 있습니다.
    
    자산총계는 "자산총계", "자산"으로 불릴 수 있습니다.
    부채총계는 "부채총계", "부채"로 불릴 수 있습니다.
    자본총계는 "자본총계", "자본", "순자산", "자본계"로 불릴 수 있습니다.
    자본금은 "자본금", "보통주자본금", "납입자본"으로 불릴 수 있습니다.
    매출액은 "외부고객으로부터의수익", "매출액", "영업수익", "수익매출액", "매출", "영업수익매출액", "매출액주석", "매출및지분법손익"으로 불릴 수 있습니다.
    영업이익은 "영업이익", "영업이익손실", "영업손실"로 불릴 수 있습니다.
    법인세차감전순이익은 "법인세차감전계속영업이익", "계속영업법인세비용차감전순이익", "법인세비용차감전계속사업이익", "법인세비용차감전계속영업이익", "법인세비용차감전손익", "법인세비용차감전순이익", "법인세차감전순이익", "법인세비용차감전순손실", "법인세비용차감전순손익", "법인세비용차감전당기순이익", "법인세비용차감전계속영업순이익", "법인세비용차감전이익", "법인세차감전순손실"로 불릴 수 있습니다.
    당기순이익은 "당기순이익", "연결당기순이익", "당기순손실", "당기순손익", "당기연결순이익"으로 불릴 수 있습니다.

    텍스트 데이터:
    {extracted_text}
    """

    try:
        # GPT-4 모델에게 직접 프롬프트를 보내고 응답을 받습니다.
        response = llm(prompt)

        # 응답이 코드 블록으로 감싸져 있을 수 있으므로 이를 제거합니다.
        response_text = response.content.strip('```json').strip('```').strip()

        # 응답을 JSON 형태로 파싱하여 리스트로 반환합니다.
        extracted_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"GPT 응답 파싱 실패: {response_text}, 에러: {e}")
        extracted_data = []
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        extracted_data = []

    return extracted_data






# Step 1: DART 공시 보고서 리스트를 리턴하는 함수
def get_dart_report_list(corp_code: str):
    # API 요청 URL
    # print("get_dart_report_list def start")
    url = 'https://opendart.fss.or.kr/api/list.json'
    all_reports = []
    pblntf_types = ['A','E','F']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2026, 12, 31)

    while start_date < end_date:
        bgn_de = start_date.strftime('%Y%m%d')
        end_de = (start_date + timedelta(days=2*365)).strftime('%Y%m%d')
        if end_de > end_date.strftime('%Y%m%d'):
            end_de = end_date.strftime('%Y%m%d')

        for pblntf_ty in pblntf_types:
            params = {
                'crtfc_key': crtfc_key,
                'corp_code': corp_code,
                'bgn_de': bgn_de,
                'end_de': end_de,
                'pblntf_ty': pblntf_ty,
                'page_no': '1',
                'page_count': '100',
                'sort': 'date',
                'sort_mth': 'desc'
            }

            response = http_requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '000':
                    for report in data['list']:
                        report_info = {
                            'corp_code': report.get('corp_code'),
                            'report_nm': report.get('report_nm'),
                            'rcept_no': report.get('rcept_no'),
                            'flr_nm': report.get('flr_nm'),
                            'rcept_dt': report.get('rcept_dt')
                        }
                        all_reports.append(report_info)
                # print(all_reports)
            else:
                print(f"HTTP 요청 실패: {response.status_code}")

        start_date += timedelta(days=2*365)

    # print("조회된 보고서 목록:")
    # for report in all_reports:
        # print(f"회사코드: {report['corp_code']}, 보고서명: {report['report_nm']}, 접수번호: {report['rcept_no']}, 제출인명: {report['flr_nm']}, 접수일자: {report['rcept_dt']}")
        
    return all_reports


# Step 2: 최신 보고서 접수번호를 리턴하는 함수
def get_latest_report_receipt_no(all_reports, baseYear, valid_receipt_no):
    latest_receipt_no = None
    secondary_receipt_no = None
    

    if valid_receipt_no is not None:
        all_reports = [report for report in all_reports if report['rcept_no'] != valid_receipt_no]

    
    # 보고서 이름에 기재정정 여부를 제외한 기본 패턴 정의
    business_report_name_pattern = f"사업보고서.*{baseYear}.*12"
    consolidated_audit_report_name_pattern = f"연결감사보고서.*{baseYear}.*12"
    audit_report_name_pattern = f"감사보고서.*{baseYear}.*12"

    for report in all_reports:
        # 대괄호로 둘러싸인 접두어 제거 및 공백 제거
        report_name = re.sub(r'\[.*?\]', '', report['report_nm']).replace(' ', '').strip()
        print(f"Checking report_name: {report_name}")  # 디버깅을 위한 출력

        if re.search(business_report_name_pattern, report_name) or re.search(consolidated_audit_report_name_pattern, report_name):
            if latest_receipt_no is None or report['rcept_no'] > latest_receipt_no:
                latest_receipt_no = report['rcept_no']
        elif re.search(audit_report_name_pattern, report_name):
            if secondary_receipt_no is None or report['rcept_no'] > secondary_receipt_no:
                secondary_receipt_no = report['rcept_no']

    return latest_receipt_no if latest_receipt_no else secondary_receipt_no


# Step 3: DART 사이트에서 보고서를 추출하는 함수
def fetch_and_create_urls(report_numbers, tryNum):
    url_base = 'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='


    Processingurl = f"{url_base}{report_numbers}"
    # print(f"Processing URL: {Processingurl}")

    response = http_requests.get(Processingurl)
    if response.status_code == 200:

        return create_report_url(response.text, report_numbers, tryNum), Processingurl
    else:
        print(f"HTTP 요청 실패: {response.status_code}")
        return None

def extract_account_data(soup):
    extracted_data = []
    unit = 1  # 기본 단위를 원으로 초기화
    unit_found = False  # 단위가 추출되었는지 확인하는 플래그

    # 모든 테이블을 처리하지만 최대 20개의 테이블만 처리
    tables = soup.find_all('table')[:20]

    for table in tables:
        
        thead = table.find('thead')
        
        if thead:
            print("thead found")    
            annotation_index = None
            th_tags = thead.find_all('th')
            for index, th in enumerate(th_tags):
                addtext = th.get_text(strip=True).replace(' ', '')
                print("index", addtext) 
                if '주석' in addtext:
                    annotation_index = index
                    print("annotation_index", annotation_index)  
                    break

            # 주석 열을 제거
            if annotation_index is not None:
                for tr in table.find_all('tr'):
                    td_tags = tr.find_all('td')
                    if len(td_tags) > annotation_index:
                        td_tags[annotation_index].decompose()  # 주석 열 태그를 실제로 HTML에서 제거

        for tr in table.find_all('tr'):
            td_tags = tr.find_all('td')

            # "단위:"라는 키워드가 포함된 <td> 태그에서 단위를 한 번만 추출
            if not unit_found:
                for td in td_tags:
                    td_text = td.get_text(strip=True)
                    if '단위' in td_text:
                        if '십억원' in td_text:
                            unit = 1000000000
                        elif '억원' in td_text:
                            unit = 100000000
                        elif '백만원' in td_text:
                            unit = 1000000
                        elif '천원' in td_text:
                            unit = 1000
                        elif '원' in td_text:
                            unit = 1
                        unit_found = True
                        break  # 단위를 찾았으므로 더 이상 반복할 필요 없음

            # <TD> 태그 안에 텍스트나 숫자 없이 태그 정보만 있는 경우 해당 태그 삭제
            for td in td_tags:
                if not td.get_text(strip=True):  # 텍스트가 없으면
                    td.decompose()  # 태그를 삭제

            # 모든 <td> 값이 비어있거나 ' ' 또는 특수문자(예: '-')로만 이루어진 경우, 해당 <td>를 삭제
            td_tags = [td for td in td_tags if re.sub(r'[\s\-]+', '', td.get_text(strip=True)) != '']

            # 모든 <td> 값이 삭제되어 비어있는 <tr>은 삭제 (continue로 건너뜀)
            if len(td_tags) <= 1:
                continue

            current_amount = None  # 초기값 설정
            account_name = None

            # 특정 키워드가 포함된 텍스트가 나오면 모든 프로세스를 중단하고 데이터 반환
            for td in td_tags:
                td_text = td.get_text(strip=True).replace(' ', '')
                if any(keyword in td_text for keyword in ['영업활동으로인한현금흐름', '희석주당이익', '배당금의지급', '기초자본']):
                    print(f"Keyword '{td_text}' found. Stopping the process.")
                    return extracted_data

            # 각 TD의 내용을 출력
            td1 = td_tags[0].get_text(strip=True) if len(td_tags) > 0 else None
            td2 = td_tags[1].get_text(strip=True) if len(td_tags) > 1 else None
            td3 = td_tags[2].get_text(strip=True) if len(td_tags) > 2 else None
            td4 = td_tags[3].get_text(strip=True) if len(td_tags) > 3 else None

            print(f"TD1: {td1}, TD2: {td2}, TD3: {td3}, TD4: {td4}")

            # 첫 번째 <td>에서 계정명을 추출
            if len(td_tags) >= 1:
                account_name = clean_account_name(td_tags[0].get_text(strip=True))
                # 계정명을 첫 번째 <td>에서 추출한 경우, current_amount는 두 번째 <td>에서 추출
                if len(td_tags) > 1:
                    current_amount = ''.join(list(td_tags[1].stripped_strings)).replace(',', '').replace('=', '')

            # 첫 번째 <td>가 공란이면 두 번째 <td>에서 계정명을 추출
            if not account_name and len(td_tags) > 1:
                account_name = clean_account_name(td_tags[1].get_text(strip=True))
                # 계정명을 두 번째 <td>에서 추출한 경우, current_amount는 세 번째 <td>에서 추출
                if len(td_tags) > 2:
                    current_amount = ''.join(list(td_tags[2].stripped_strings)).replace(',', '').replace('=', '')

            # account_name과 current_amount를 출력
            print(f"Account Name: {account_name}, Current Amount: {current_amount}")

            # 금액에서 괄호를 제거하고 음수로 변환
            if current_amount:
                current_amount = current_amount.replace(',', '')
                if current_amount.startswith('(') and current_amount.endswith(')'):
                    current_amount = '-' + current_amount[1:-1]

                try:
                    current_amount = int(current_amount) * unit  # 단위를 곱하여 원단위로 변환
                except ValueError:
                    current_amount = None  # 금액이 숫자가 아닐 경우 None 처리

            # 계정명과 당기 금액을 딕셔너리로 저장
            if account_name and current_amount is not None:
                extracted_data.append({
                    'account_name': account_name,
                    'current_amount': current_amount
                })

    return extracted_data




def clean_account_name(name: str) -> str:
    # 계정명 정제를 위한 공통 함수
    name = re.sub(r'\(주\d{1,3}(,\d{1,3})*\)', '', name)
    name = re.sub(r'\(주석\s*\d{1,3}(,\s*\d{1,3})*\)', '', name)
    name = re.sub(r'\((손실|순손실|이익)\)', '', name)
    name = re.sub(r'\(단위\s*:\s*[가-힣]+\)', '', name)
    name = re.sub(r'[^가-힣]', '', name)
    return name


# 기타 함수들
def create_report_url(html_content, rcp_no, tryNum):
    # 우선순위에 따라 키워드 목록을 정의합니다.
    # print(html_content)
    if tryNum == 1:
        keyword_patterns = [
            r"연\s*결\s*재\s*무\s*제\s*표",
            r"연\s*결\s*재\s*무\s*상\s*태\s*표",
            r"재\s*무\s*제\s*표",
            r"재\s*무\s*상\s*태\s*표"
        ]
    elif tryNum == 2:
        # '연결'이라는 키워드가 포함되지 않은 패턴만 사용
        keyword_patterns = [
            r"재\s*무\s*제\s*표",
            r"재\s*무\s*상\s*태\s*표"
        ]

    lines = html_content.split('\n')
    node_data = {}
    
    for pattern in keyword_patterns:
        keyword_regex = re.compile(pattern)
        
        for i, line in enumerate(lines):
            # '연결'이 포함된 줄은 제외
            if '연결' in line and tryNum == 2:
                continue
            
            if keyword_regex.search(line):
                node_data = extract_node_data(lines, i)
                if node_data:
                    report_url = format_report_url(node_data, rcp_no)
                    print(f"report_url detail: {report_url}")
                    
                    
                    return fetch_and_save_report_details(report_url) #report_url을 바탕으로 SOUP리턴

    return None

def extract_node_data(lines, start_index):
    node_data = {}
    patterns = {
        'dcmNo': r"node\d+\['dcmNo'\]\s*=\s*\"(\d+)\";",
        'eleId': r"node\d+\['eleId'\]\s*=\s*\"(\d+)\";",
        'offset': r"node\d+\['offset'\]\s*=\s*\"(\d+)\";",
        'length': r"node\d+\['length'\]\s*=\s*\"(\d+)\";"
    }

    # 각 값을 처음으로 매칭된 패턴에 따라 추출합니다.
    for key, pattern in patterns.items():
        for line in lines[start_index:start_index + 100]:  # 범위를 100줄로 제한
            match = re.search(pattern, line)
            if match:
                node_data[key] = match.group(1)
                break  # 첫 번째로 매칭된 값만 사용
                
    return node_data

def format_report_url(data, rcp_no):
    # 필요한 모든 데이터가 존재하는지 확인
    if all(k in data for k in ('dcmNo', 'eleId', 'offset', 'length')):
        return f"https://dart.fss.or.kr/report/viewer.do?rcpNo={rcp_no}&dcmNo={data['dcmNo']}&eleId={data['eleId']}&offset={data['offset']}&length={data['length']}&dtd=dart4.xsd"
    else:
        return None
    
def fetch_and_save_report_details(report_url):
    response = http_requests.get(report_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # print(f"fetch_and_save_report_details SOUP: {soup}")
        return soup
    else:
        print(f"HTTP 요청 실패: {response.status_code}")
        return None


def FSmaker(extracted_data, url):
    fs_dict = {
        "자산총계": None,
        "부채총계": None,
        "자본총계": None,
        "자본금": None,
        "매출액": None,
        "영업이익": None,
        "법인세차감전순이익": None,
        "당기순이익": None,
        "재무제표url" : url,
        "재무제표상세" : extracted_data
    }
    
    # 필요한 항목 이름 매핑 (복수의 항목이 매핑될 수 있음)
    keys_mapping = {
        "자산총계": ["자산총계", "자산"],
        "부채총계": ["부채총계", "부채"],
        "자본총계": ["자본총계","자본","순자산","자본계"],
        "자본금": ["자본금", "보통주자본금","납입자본"],
        "매출액": ["외부고객으로부터의수익", "매출액", "영업수익", "수익매출액", "매출", "영업수익매출액","매출액주석","매출및지분법손익"],
        "영업이익": ["영업이익", "영업이익손실", "영업손실"],
        "법인세차감전순이익": ["법인세차감전계속영업이익","계속영업법인세비용차감전순이익","법인세비용차감전계속사업이익","법인세비용차감전계속영업이익","법인세비용차감전손익", "법인세비용차감전순이익", "법인세차감전순이익", "법인세비용차감전순손실", "법인세비용차감전순손익", "법인세비용차감전당기순이익","법인세비용차감전계속영업순이익","법인세비용차감전이익","법인세차감전순손실"],
        "당기순이익": ["당기순이익", "당기순이익", "연결당기순이익", "당기순손실","당기순손익","당기연결순이익", "당기연결총포괄이익"]
    }

    # 매출액 관련 항목
    revenue_sources = {
        "보험영업수익": None,
        "투자영업수익": None,
        "이자수익": None,
        "수수료수익": None
    }
    
    # 데이터에서 항목 추출
    for item in extracted_data:
        account_name = item['account_name']
        current_amount = item['current_amount']
        
        # current_amount가 None인 경우 무시하고 다음 항목으로 넘어감
        if current_amount is None:
            continue
        
        # 손실 항목일 경우 음수로 변환
        if account_name.endswith("손실"):
            current_amount = -abs(current_amount)

        # 기본 항목 추출 (복수 항목 매핑 가능)
        for key, names in keys_mapping.items():
            for name in names:
                if name == account_name and (fs_dict[key] is None or fs_dict[key] == 0):  # 현재 None 또는 0인 경우에도 반영
                    fs_dict[key] = current_amount
                    break  # 값을 찾았으면 다른 매핑 항목은 볼 필요 없음

        # 매출액 관련 항목 추출 (정확한 일치만 허용)
        for key in revenue_sources.keys():
            if key == account_name:  # 정확히 일치하는 경우만 처리
                if revenue_sources[key] is None:
                    revenue_sources[key] = current_amount
                    break  # 최초의 일치하는 값을 반영하면 루프 종료
                
    # 매출액 계산
    if fs_dict["매출액"] is None:
        # 보험영업수익, 투자영업수익을 합산하여 매출액으로 사용
        if revenue_sources["보험영업수익"] is not None:
            fs_dict["매출액"] = (
                (revenue_sources["보험영업수익"] or 0) +
                (revenue_sources["투자영업수익"] or 0)
            )
        # 보험영업수익이 없을 경우 이자수익과 수수료 수익을 합산하여 매출액으로 사용
        elif revenue_sources["이자수익"] is not None:
            fs_dict["매출액"] = (
                (revenue_sources["이자수익"] or 0) +
                (revenue_sources["수수료수익"] or 0)
            )

    return fs_dict



def fs_db_insert(db: Session, jurir_no: str, fs_dict: dict, baseYear: int):
    try:
        # 테이블 이름을 연도에 따라 동적으로 설정
        table_name = f"FS{baseYear}"

        # MetaData를 생성하고 테이블을 동적으로 참조
        metadata = MetaData()
        fs_table = Table(table_name, metadata, autoload_with=db.bind)
        
        # 먼저 해당 jurir_no의 데이터를 삭제
        db.execute(fs_table.delete().where(fs_table.c.jurir_no == jurir_no))

        # 계산된 필드들 (None일 경우 0으로 대체)
        total_asset = fs_dict.get("자산총계") or 0
        total_debt = fs_dict.get("부채총계") or 0
        total_equity = fs_dict.get("자본총계") or 0
        revenue = fs_dict.get("매출액") or 0
        net_income = fs_dict.get("당기순이익") or 0
        capital = fs_dict.get("자본금") or 0
        operating_income = fs_dict.get("영업이익") or 0
        earning_before_tax = fs_dict.get("법인세차감전순이익") or 0
        fs_url = fs_dict.get("재무제표url")
        # JSON 데이터를 인코딩할 때 ensure_ascii=False 옵션을 추가
        fs_detail = json.dumps(fs_dict.get("재무제표상세"), ensure_ascii=False)

        # div/0 같은 에러가 발생할 수 있는 계산은 예외 처리를 추가하여 0으로 대체
        try:
            debt_ratio = round(total_debt / total_equity * 100, 2) if total_equity != 0 else 0
        except ZeroDivisionError:
            debt_ratio = 0

        try:
            margin = round(net_income / revenue, 3) if revenue != 0 else 0
        except ZeroDivisionError:
            margin = 0

        try:
            turnover = round(revenue / total_asset, 3) if total_asset != 0 else 0
        except ZeroDivisionError:
            turnover = 0

        try:
            leverage = round(total_asset / total_equity, 3) if total_equity != 0 else 0
        except ZeroDivisionError:
            leverage = 0
        
        base_date = datetime(baseYear, 12, 31)  # baseYear + 12 + 31을 날짜 형식으로

        # 삽입할 데이터 생성
        insert_data = {
            'baseDate': base_date,  # baseDate
            'bizYear': baseYear,  # bizYear
            'jurir_no': jurir_no,  # jurir_no
            'currency': 'KRW',  # currency
            'fsCode': fs_dict.get("fsCode"),  # fsCode
            'fsName': fs_dict.get("fsName"),  # fsName
            f'totalAsset{baseYear}': total_asset,  # totalAsset{baseYear}
            f'totalDebt{baseYear}': total_debt,  # totalDebt{baseYear}
            f'totalEquity{baseYear}': total_equity,  # totalEquity{baseYear}
            f'capital{baseYear}': capital,  # capital{baseYear}
            f'revenue{baseYear}': revenue,  # revenue{baseYear}
            f'operatingIncome{baseYear}': operating_income,  # operatingIncome{baseYear}
            f'earningBeforeTax{baseYear}': earning_before_tax,  # earningBeforeTax{baseYear}
            f'netIncome{baseYear}': net_income,  # netIncome{baseYear}
            f'debtRatio{baseYear}': debt_ratio,  # debtRatio{baseYear}
            f'margin{baseYear}': margin,  # margin{baseYear}
            f'turnover{baseYear}': turnover,  # turnover{baseYear}
            f'leverage{baseYear}': leverage,  # leverage{baseYear}
            'created_at': datetime.now(),  # created_at (timestamp)
            'FS_url': fs_url,  # FS_url
            'FS_detail': fs_detail  # FS_detail
        }

        # 새로운 레코드 삽입
        db.execute(fs_table.insert().values(insert_data))

        # 변경 사항을 커밋하여 DB에 반영
        db.commit()

    except Exception as e:
        db.rollback()  # 오류가 발생한 경우 롤백
        print(f"An error occurred while inserting/updating {table_name}: {str(e)}")
        raise


def get_company_infoFS_list(db: Session, jurir_no_list: List[str]):
    query = db.query(
        CompanyInfo.corp_code,
        CompanyInfo.corp_name,
        CompanyInfo.corp_name_eng,
        CompanyInfo.stock_name,
        CompanyInfo.stock_code,
        CompanyInfo.ceo_nm,
        CompanyInfo.corp_cls,
        CompanyInfo.jurir_no.label("company_jurir_no"),
        CompanyInfo.bizr_no,
        CompanyInfo.adres,
        CompanyInfo.hm_url,
        CompanyInfo.ir_url,
        CompanyInfo.phn_no,
        CompanyInfo.fax_no,
        CompanyInfo.induty_code,
        CompanyInfo.est_dt,
        CompanyInfo.acc_mt,
        FS2023.id,
        FS2023.baseDate,
        FS2023.bizYear,
        FS2023.currency,
        FS2023.fsCode,
        FS2023.fsName,
        cast(FS2023.totalAsset2023 / 100000000, Integer).label('totalAsset2023'),
        cast(FS2023.totalDebt2023 / 100000000, Integer).label('totalDebt2023'),
        cast(FS2023.totalEquity2023 / 100000000, Integer).label('totalEquity2023'),
        cast(FS2023.capital2023 / 100000000, Integer).label('capital2023'),
        cast(FS2023.revenue2023 / 100000000, Integer).label('revenue2023'),
        cast(FS2023.operatingIncome2023 / 100000000, Integer).label('operatingIncome2023'),
        cast(FS2023.earningBeforeTax2023 / 100000000, Integer).label('earningBeforeTax2023'),
        cast(FS2023.netIncome2023 / 100000000, Integer).label('netIncome2023'),
        FS2023.debtRatio2023,
        FS2023.margin2023,
        FS2023.turnover2023,
        FS2023.leverage2023,
        FS2023.created_at
    ).join(FS2023, CompanyInfo.jurir_no == FS2023.jurir_no).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()
    
    return query

def get_company_info_list(db: Session, jurir_no_list: List[str]) -> List[CompanyInfo]:
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()

def get_company_info_list(db: Session, jurir_no_list: List[str]) -> List[CompanyInfo]:
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no.in_(jurir_no_list)).all()

def get_company_info(db: Session, jurir_no: str):
    return db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()


def get_company_info2(db: Session, corp_code: str):
    return db.query(CompanyInfo).filter(CompanyInfo.corp_code == corp_code).first()

def get_Stock_data(db: Session, corp_code: str):
    stock_data = db.query(StockData).filter(StockData.corp_code == corp_code).first()
    if stock_data is None:
        # Return a default object with all attributes set to 0
        return StockData(
            market_capitalization=0,
            per_value=0,
            pbr_value=0,
            cagr_1y=0,
            cagr_3y=0,
            cagr_5y=0,
            vol_1y=0,
            vol_3y=0,
            vol_5y=0
            # Add other fields if needed and set them to 0
        )
    return stock_data

def get_FS2023_list(db: Session, jurir_no_list: List[str]) -> List[FS2023]:
    return db.query(FS2023).filter(FS2023.jurir_no.in_(jurir_no_list)).all()

## 총자산 크기로 뽑기
# def get_sample_jurir_no(db: Session) -> List[str]:
#     CompanyInfoAlias = aliased(CompanyInfo)
#     FS2023Alias = aliased(FS2023)

#     subquery = (
#         db.query(CompanyInfoAlias.jurir_no)
#         .join(FS2023Alias, CompanyInfoAlias.jurir_no == FS2023Alias.jurir_no)
#         .filter(FS2023Alias.totalAsset2023 > 0)
#         .order_by(FS2023Alias.totalAsset2023.desc())
#         .offset(30)  # 시작점
#         .limit(40)   # 종료점    
#         )

#     jurir_no_list = [result[0] for result in subquery.all()]
#     return jurir_no_list

## 랜덤으로 10개 뽑기 
def get_sample_jurir_no(db: Session) -> List[str]:
    # SQL 쿼리를 직접 실행하기 위해 text() 함수를 사용
    query = text("""
        SELECT a.jurir_no 
        FROM FS2023_backup a
        LEFT OUTER JOIN companyInfo b
        ON a.jurir_no = b.jurir_no
        WHERE a.totalAsset2023 > 0
        and a.jurir_no not in 
        (select jurir_no from FS2023)
        ORDER BY a.totalAsset2023 DESC
        LIMIT 10;
    """)

    # 쿼리를 실행하고 결과를 리스트로 변환
    result = db.execute(query)
    jurir_no_list = [row[0] for row in result]  # 튜플의 첫 번째 요소로 접근

    return jurir_no_list



def get_FS2023(db: Session, jurir_no: str):
    # Retrieve the data from the database2
    # print("get_FS2023 started")
    fs_data = db.query(FS2023).filter(FS2023.jurir_no == jurir_no).first()

    if fs_data:
        # If totalAsset2023 is greater than 0, return the actual data
        # print("get_FS2023 started")
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2023(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2023=0,
            totalDebt2023=0,
            totalEquity2023=0,
            capital2023=0,
            revenue2023=0,
            operatingIncome2023=0,
            earningBeforeTax2023=0,
            netIncome2023=0,
            debtRatio2023=0.0,
            margin2023=0.0,
            turnover2023=0.0,
            leverage2023=0.0,
            created_at=None
        )

def get_FS2022(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2022).filter(FS2022.jurir_no == jurir_no).first()

    if fs_data:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2022(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2022=0,
            totalDebt2022=0,
            totalEquity2022=0,
            capital2022=0,
            revenue2022=0,
            operatingIncome2022=0,
            earningBeforeTax2022=0,
            netIncome2022=0,
            debtRatio2022=0.0,
            margin2022=0.0,
            turnover2022=0.0,
            leverage2022=0.0,
            created_at=None
        )

def get_FS2021(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2021).filter(FS2021.jurir_no == jurir_no).first()

    if fs_data:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2021(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2021=0,
            totalDebt2021=0,
            totalEquity2021=0,
            capital2021=0,
            revenue2021=0,
            operatingIncome2021=0,
            earningBeforeTax2021=0,
            netIncome2021=0,
            debtRatio2021=0.0,
            margin2021=0.0,
            turnover2021=0.0,
            leverage2021=0.0,
            created_at=None

        )

def get_FS2020(db: Session, jurir_no: str):
    # Retrieve the data from the database
    fs_data = db.query(FS2020).filter(FS2020.jurir_no == jurir_no).first()

    if fs_data:
        # If totalAsset2023 is greater than 0, return the actual data
        return fs_data
    else:
        # Return a new FS2023 object with default values
        return FS2020(
            baseDate=None,
            bizYear=None,
            jurir_no=jurir_no,
            currency=None,
            fsCode=None,
            fsName=None,
            totalAsset2020=0,
            totalDebt2020=0,
            totalEquity2020=0,
            capital2020=0,
            revenue2020=0,
            operatingIncome2020=0,
            earningBeforeTax2020=0,
            netIncome2020=0,
            debtRatio2020=0.0,
            margin2020=0.0,
            turnover2020=0.0,
            leverage2020=0.0,
            created_at=None
        )

"""
def get_FS2022(db: Session, jurir_no: str):
    return db.query(FS2022).filter(FS2022.jurir_no == jurir_no).first()

def get_FS2021(db: Session, jurir_no: str):
    return db.query(FS2021).filter(FS2021.jurir_no == jurir_no).first()

def get_FS2020(db: Session, jurir_no: str):
    return db.query(FS2020).filter(FS2020.jurir_no == jurir_no).first()
"""    

def get_corp_info_code(corp_code: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.corp_code == corp_code).first()
        return selectedcompany
    finally:
        db.close()
        
def get_corp_info_name(corp_name: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.corp_name == corp_name).first()
        return selectedcompany
    finally:
        db.close()
        
def get_corp_info_jurir_no(jurir_no: str):
    db: Session = SessionLocal()
    try:
        selectedcompany = db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
        return selectedcompany
    finally:
        db.close()
        
        
def get_autocomplete_suggestions(search_type: str, query: str) -> List[str]:
    db: Session = SessionLocal()
    if search_type == "corp_name":
        results = db.query(CompanyInfo.corp_name).filter(CompanyInfo.corp_name.like(f"{query}%")).limit(5).all()
    elif search_type == "jurir_no":
        results = db.query(CompanyInfo.jurir_no).filter(CompanyInfo.jurir_no.like(f"{query}%")).limit(5).all()
    elif search_type == "corp_code":
        results = db.query(CompanyInfo.corp_code).filter(CompanyInfo.corp_code.like(f"{query}%")).limit(5).all()
    else:
        results = []

    return [result[0] for result in results]


async def get_stockgraph(stock_code: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = []
    max_pages = 9
    page = 1

    while page <= max_pages:
        url = f"https://finance.naver.com/item/sise_day.nhn?code={stock_code}&page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        
        soup = BeautifulSoup(response.text, 'html.parser')

        rows = soup.select("table.type2 tr")
        if not rows or len(rows) == 0:
            break  # 종료 조건: 더 이상 데이터가 없으면 종료

        page_data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                date_text = cols[0].get_text(strip=True)
                if date_text:
                    try:
                        date = datetime.strptime(date_text, '%Y.%m.%d').strftime('%Y-%m-%d')
                        if datetime.strptime(date, '%Y-%m-%d') >= start_date:
                            open_price = float(cols[3].get_text(strip=True).replace(',', ''))
                            high_price = float(cols[4].get_text(strip=True).replace(',', ''))
                            low_price = float(cols[5].get_text(strip=True).replace(',', ''))
                            close_price = float(cols[1].get_text(strip=True).replace(',', ''))
                            page_data.append({
                                "t": date,
                                "o": open_price,
                                "h": high_price,
                                "l": low_price,
                                "c": close_price
                            })
                    except ValueError:
                        continue

        stock_data.extend(page_data)

        # 페이지 증가
        page += 1

    stock_data.reverse()  # 데이터를 오래된 순으로 정렬
    return {"stock_data": stock_data[:90]}  # 최대 90일치 데이터만 반환

# get_stockgraph1 함수
async def get_stockgraph1(stock_code: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    # 주식 코드에 올바른 접미사 추가
    stock_code = stock_code if '.' in stock_code else f'{stock_code}.KS'  # 예: 한국 주식의 경우 ".KS" 또는 ".KQ"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    try:
        # yfinance를 사용하여 데이터 가져오기
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        # 데이터가 없는 경우 처리
        if hist.empty:
            raise ValueError(f"주가정보가 존재하지 않습니다.: {stock_code}")

        stock_data = []
        for date, row in hist.iterrows():
            stock_data.append({
                "t": date.strftime('%Y-%m-%d'),
                "o": row['Open'],
                "h": row['High'],
                "l": row['Low'],
                "c": row['Close']
            })

        return {"stock_data": stock_data}
    except Exception as e:
        # 기타 예외 처리
        print(f"오류가 발생했습니다: {e}")
        raise ValueError(f"An unexpected error occurred while retrieving data for stock code: {stock_code}") from e
    



def generate_pdf(html_content):
    path_to_wkhtmltopdf = '/usr/bin/wkhtmltox/bin/wkhtmltopdf'  # 경로를 자신의 시스템에 맞게 수정
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

    options = {
        'page-size': 'A4',
        'encoding': 'UTF-8',
        'no-outline': None,
        'no-stop-slow-scripts': None,
        'enable-local-file-access': None,
        'zoom': '0.9',  # 확대 비율을 약간 줄여 내용이 잘리지 않도록 조정
        'custom-header': [
            ('Accept-Encoding', 'gzip')
        ],
        'print-media-type': None,
        'margin-top': '5mm',
        'margin-right': '5mm',
        'margin-bottom': '5mm',
        'margin-left': '5mm',
        'disable-smart-shrinking': None,  # 스마트 축소 비활성화
    }

    pdf_path = "./static/images/Spoon_Report.pdf"


    try:
        pdfkit.from_string(html_content, pdf_path, options=options, configuration=config)
    except Exception as e:
        logging.error(f'PDF generation failed: {e}')
    return pdf_path


from sqlalchemy.exc import IntegrityError

class FavoriteService:
    def __init__(self, db: Session):
        self.db = db

    def toggle_favorite(self, username: str, corp_code: str) -> dict:
        # Check if the favorite already exists
        favorite = self.db.query(Favorite).filter_by(username=username, corp_code=corp_code).first()
        
        try:
            if favorite:
                # If exists, remove it
                self.db.delete(favorite)
                self.db.commit()
                return {"is_favorited": False}
            else:
                # If not exists, add it
                new_favorite = Favorite(username=username, corp_code=corp_code)
                self.db.add(new_favorite)
                self.db.commit()
                return {"is_favorited": True}
        except Exception as e:  # Generic exception handling
            # Rollback the session if an error occurs
            self.db.rollback()
            # Log the error (could also use a logging framework)
            print(f"Error: {e}")
            # Handle the specific error (optional)
            raise

    def is_favorite(self, username: str, corp_code: str) -> bool:
        return self.db.query(Favorite).filter_by(username=username, corp_code=corp_code).first() is not None

    def get_favorites_for_user(self, username: str):
        return self.db.query(Favorite).filter_by(username=username).all()
    
    
def get_username_from_session(request: Request):
    username = request.session.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )       
    return username

def get_favorite_companies(db: Session, username: str):
    favorites = db.query(Favorite).filter(Favorite.username == username).all()
    favorite_companies = []
    for favorite in favorites:
        company_info = db.query(CompanyInfo).filter(CompanyInfo.corp_code == favorite.corp_code).first()
        if company_info:
            financial_info = db.query(FS2023).filter(FS2023.jurir_no == company_info.jurir_no).first()
            favorite_companies.append({
                "corp_code": company_info.corp_code,
                "jurir_no": company_info.jurir_no,
                "corp_name": company_info.corp_name,
                "ceo_nm": company_info.ceo_nm,
                "corp_cls": company_info.corp_cls,
                "totalAsset2023": financial_info.totalAsset2023 // 100000000 if financial_info else None,
                "capital2023": financial_info.capital2023 // 100000000 if financial_info else None,
                "revenue2023": financial_info.revenue2023 // 100000000 if financial_info else None,
                "netIncome2023": financial_info.netIncome2023 // 100000000 if financial_info else None,

            })
    return favorite_companies

def add_recent_view(db: Session, username: str, corp_code: str):
    # Check if the recent view already exists
    existing_view = db.query(RecentView).filter(RecentView.username == username, RecentView.corp_code == corp_code).first()
    
    if existing_view:
        # Update the timestamp if the view exists
        existing_view.created_at = datetime.utcnow()
    else:
        # Add a new view if it doesn't exist
        new_view = RecentView(username=username, corp_code=corp_code, created_at=datetime.utcnow())
        db.add(new_view)

    # Remove the oldest view if more than 5 views are present
    recent_views = db.query(RecentView).filter(RecentView.username == username).order_by(RecentView.created_at.desc()).all()
    if len(recent_views) > 5:
        oldest_view = recent_views[-1]
        db.delete(oldest_view)
    
    db.commit()

    return existing_view if existing_view else new_view

def get_recent_views(db: Session, username: str):
    recent_views = db.query(RecentView).filter(RecentView.username == username).order_by(RecentView.created_at.desc()).limit(5).all()
    
    recent_views_companies = []
    for view in recent_views:
        company_info = db.query(CompanyInfo).filter(CompanyInfo.corp_code == view.corp_code).first()
        if company_info:
            financial_info = db.query(FS2023).filter(FS2023.jurir_no == company_info.jurir_no).first()
            recent_views_companies.append({
                "corp_code": company_info.corp_code,
                "jurir_no": company_info.jurir_no,
                "corp_name": company_info.corp_name,
                "corp_code": company_info.corp_code,
                "ceo_nm": company_info.ceo_nm,
                "corp_cls": company_info.corp_cls,
                "totalAsset2023": (financial_info.totalAsset2023 // 100000000) if financial_info and financial_info.totalAsset2023 else None,
                "capital2023": (financial_info.capital2023 // 100000000) if financial_info and financial_info.capital2023 else None,
                "revenue2023": (financial_info.revenue2023 // 100000000) if financial_info and financial_info.revenue2023 else None,
                "netIncome2023": (financial_info.netIncome2023 // 100000000) if financial_info and financial_info.netIncome2023 else None,
            })
    return recent_views_companies






