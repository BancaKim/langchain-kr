from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
password = urllib.parse.quote_plus("!Q@W3e4r")  # 특수 문자를 URL 인코딩
# 동기
DATABASE_URL = f"mysql+pymysql://manager:{password}@211.37.179.178/spoon"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

crtfc_key = '1ac7d99734144b014e528c977f5c6a1c9831d76c'  # 여기에 본인의 API 인증키를 입력하세요