import os
import smtplib
from email.mime.text import MIMEText
from fastapi import BackgroundTasks
from dotenv import load_dotenv

load_dotenv()

NAVER_EMAIL = os.getenv("NAVER_EMAIL")
NAVER_PASSWORD = os.getenv("NAVER_PASSWORD")

def send_email(background_tasks: BackgroundTasks, subject: str, to: str, body: str):
    def email_task():
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = NAVER_EMAIL
        msg['To'] = 'oo5450@naver.com'

        with smtplib.SMTP_SSL('smtp.naver.com', 465) as server:
            server.login(NAVER_EMAIL, NAVER_PASSWORD)
            server.sendmail(msg['From'], [msg['To']], msg.as_string())

    background_tasks.add_task(email_task)
