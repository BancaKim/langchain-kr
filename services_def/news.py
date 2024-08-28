import requests
from typing import List, Dict
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import re

# .env 파일에서 환경 변수 로드
load_dotenv()

# 네이버 API 크리덴셜을 환경 변수에서 가져옴
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

def fetch_naver_news(corporation_name: str) -> List[Dict]:
    """
    네이버 뉴스 API를 사용하여 특정 법인명과 관련된 최신 뉴스를 가져옵니다.
    
    Args:
        corporation_name (str): 검색할 법인명.
        
    Returns:
        List[Dict]: 뉴스 기사 목록.
    """
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        'X-Naver-Client-Id': client_id,
        'X-Naver-Client-Secret': client_secret
    }
    
    # 검색어 전처리: 특수문자 제거
    processed_name = re.sub(r'[\(\)주]', '', corporation_name)  # 괄호와 '(주)' 제거
    
    params = {
        'query': processed_name,
        'display': 100,  # 100개 뉴스를 가져와 필터링
        'sort': 'date'  # 최신 뉴스부터 정렬
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        news_items = response.json().get('items', [])
        # 제목에 법인명이 포함된 뉴스만 필터링
        filtered_news = [item for item in news_items if processed_name.lower() in item['title'].lower()]
        
        # 필터링된 뉴스가 없으면 전체 뉴스에서 상위 5개 반환
        if not filtered_news:
            return news_items[:5]
        # 두 줄 추가됨, 제목만 하려면 삭제
        return filtered_news[:5]  # 가장 관련성이 높은 5개 뉴스 반환
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch news: {response.reason}")


