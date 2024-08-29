# app/models/data_model.py
from pydantic import BaseModel
from sqlalchemy import Column, String, Float, DateTime, Integer, LargeBinary, JSON, Boolean, Integer
from database import Base
from datetime import datetime

from sqlalchemy.ext.declarative import declarative_base

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
    
    
    


Base = declarative_base()

class CompanyInfo(Base):
    __tablename__ = 'companyInfo'
    
    corp_code = Column(String(8), primary_key=True)
    corp_name = Column(String(255), nullable=False)
    corp_name_eng = Column(String(255))
    stock_name = Column(String(255))
    stock_code = Column(String(6))
    ceo_nm = Column(String(255))
    corp_cls = Column(String(1))
    jurir_no = Column(String(13), unique=True)
    bizr_no = Column(String(13))
    adres = Column(String(255))
    hm_url = Column(String(255))
    ir_url = Column(String(255))
    phn_no = Column(String(20))
    fax_no = Column(String(20))
    induty_code = Column(String(10))
    est_dt = Column(String(8))
    acc_mt = Column(String(2))

    def __repr__(self):
        return f"<CompanyInfo(corp_code={self.corp_code}, corp_name={self.corp_name})>"


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