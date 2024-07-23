from sqlalchemy import Column, Integer, Numeric, String, Float, Date
from database import Base

## dart 기준 회사 정보
class CompanyInfo(Base):
    __tablename__ = 'companyInfo'

    corp_code = Column(String(8), primary_key=True)
    corp_name = Column(String(255))
    corp_name_eng = Column(String(255))
    stock_name = Column(String(255))
    stock_code = Column(String(6))
    ceo_nm = Column(String(255))
    corp_cls = Column(String(1))
    jurir_no = Column(String(13))
    bizr_no = Column(String(13))
    adres = Column(String(255))
    hm_url = Column(String(255))
    ir_url = Column(String(255))
    phn_no = Column(String(20))
    fax_no = Column(String(20))
    induty_code = Column(String(10))
    est_dt = Column(String(8))
    acc_mt = Column(String(2))