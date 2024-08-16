# sms.py

import logging
import os
from pickle import FALSE, TRUE
from dotenv import load_dotenv
from fastapi import logger
from jinja2 import Environment, FileSystemLoader, Template
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException
from sqlalchemy import ARRAY

load_dotenv()
env = Environment(loader=FileSystemLoader('templates/contact'))  # 템플릿 디렉토리 설정
# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_sms(username, corporation_name, content):
    api_key = os.getenv("SMS_API_KEY")
    api_secret = os.getenv("SMS_API_SECRET")
    from_number = os.getenv("FROM_NUMBER")
    to_number = os.getenv("TO_NUMBER")
    url = os.getenv("DETAIL_URL")  # URL도 .env에서 불러오기
    
    # HTML 템플릿 로드 및 렌더링
    template = env.get_template('sms_template.txt')
    message = template.render(username=username, corporation_name=corporation_name, content=content, url=url)
    

    params = {
        'type': 'lms',   #sms, lms, mms 선택
        'to': to_number,
        'from': from_number,  # 발신자 번호 (인증된 번호만 가능)
        'text': message
    }

    cool = Message(api_key, api_secret, True)
    
    print(cool)

    print(to_number)
    print(from_number)
    print(message)
    print(api_key)
    print(api_secret)    

    logger.info("Preparing to send SMS...")
    logger.debug(f"Parameters: to={to_number}, from={from_number}, message={message}")

    try:
        response = cool.send(params)
        logger.info(f"SMS sent successfully. Response: {response}")
        return {
            "success_count": response['success_count'],
            "error_count": response['error_count'],
            "group_id": response.get('group_id'),
            "error_list": response.get('error_list', [])
        }
    except CoolsmsException as e:
        logger.error(f"CoolsmsException occurred: {e.code} - {e.msg}")
        return {
            "error_code": e.code,
            "error_message": e.msg
        }
    except Exception as e:
        logger.exception(f"Unexpected exception occurred: {str(e)}")
        return {
            "error_code": "UNEXPECTED_ERROR",
            "error_message": str(e)
        }
