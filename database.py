from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
password = urllib.parse.quote_plus("!Q@W3e4r") # 특수 문자를 URL 인코딩
#동기
DATABASE_URL = f"mysql+pymysql://manager:{password}@211.37.179.178/spoon"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()