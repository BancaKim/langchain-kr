import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import BackgroundTasks
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from models.common_models import User

load_dotenv()

NAVER_EMAIL = os.getenv("NAVER_EMAIL")
NAVER_PASSWORD = os.getenv("NAVER_PASSWORD")

env = Environment(loader=FileSystemLoader('templates'))  # 템플릿 디렉토리 설정

def send_email(background_tasks: BackgroundTasks, subject: str, to: str, template_name: str, context: dict):
    def email_task():
        # HTML 템플릿 로드 및 렌더링
        template = env.get_template(template_name)
        body = template.render(context)

        msg = MIMEMultipart("alternative")
        msg['Subject'] = subject
        msg['From'] = NAVER_EMAIL
        msg['To'] = to  # 동적으로 설정된 이메일 주소
        
        # HTML 이메일 내용 추가
        part = MIMEText(body, "html")
        msg.attach(part)

        with smtplib.SMTP_SSL('smtp.naver.com', 465) as server:
            server.login(NAVER_EMAIL, NAVER_PASSWORD)
            server.sendmail(msg['From'], [msg['To']], msg.as_string())

    background_tasks.add_task(email_task)

def find_supervisor_email(db: Session, region_headquarter_name: str):
    supervisor = db.query(User).filter(User.region_headquarter_name == region_headquarter_name, User.position_name == '본부장').first()
    if supervisor:
        return supervisor.email
    return None
