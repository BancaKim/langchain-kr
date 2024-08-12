# app/models/data_model.py
from pydantic import BaseModel
from sqlalchemy import Column, String, Float, DateTime, Integer, LargeBinary, JSON, Boolean
from database import Base
from datetime import datetime

class ModelStorage(Base):
    __tablename__ = "model_storage"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), unique=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    creation_date = Column(DateTime, nullable=False)
    n_estimators = Column(Integer)
    max_features = Column(String(255))
    accuracy = Column(Float)
    feature_importances = Column(JSON)
    n_samples = Column(Integer)
    model_filepath = Column(String(500), nullable=False)
    feature_columns = Column(JSON)
    scaler = Column(LargeBinary)
    class_report = Column(JSON)
    conf_matrix = Column(JSON)
    is_default = Column(Boolean, default=False)  # 디폴트 여부 필드 추가

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