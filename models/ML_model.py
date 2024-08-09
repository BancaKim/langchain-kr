# app/models/data_model.py
from pydantic import BaseModel
from sqlalchemy import Column, String, Float, DateTime, Integer, LargeBinary
from database import Base
from datetime import datetime

class ModelInfo(Base):
    __tablename__ = "model_info"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), index=True)  # VARCHAR 길이 지정
    creation_date = Column(DateTime, default=datetime.utcnow)
    n_estimators = Column(Integer)
    max_features = Column(String(255))  # VARCHAR 길이 지정
    accuracy = Column(Float)
    feature_importances = Column(String)  # JSON 형태로 저장할 수 있음
    n_samples = Column(Integer)
    model_binary = Column(LargeBinary)  # 모델 객체를 직렬화하여 저장
    feature_columns = Column(String)  # 피처 컬럼 이름을 JSON으로 저장
   


# class DataModel(BaseModel):
#     loan: float
#     IR: float
#     asset2023: float
#     debt2023: float
#     equity2023: float
#     revenue2023: float
#     operatingincome2023: float
#     EBT2023: float
#     margin2023: float
#     turnover2023: float
#     leverage2023: float
#     asset2022: float
#     debt2022: float
#     equity2022: float
#     revenue2022: float
#     operatingincome2022: float
#     EBT2022: float
#     margin2022: float
#     turnover2022: float
#     leverage2022: float
#     asset2021: float
#     debt2021: float
#     equity2021: float
#     revenue2021: float
#     operatingincome2021: float
#     EBT2021: float
#     margin2021: float
#     turnover2021: float
#     leverage2021: float