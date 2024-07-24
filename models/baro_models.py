from sqlalchemy import Column, Integer, Numeric, String, Float, Date, BigInteger, TIMESTAMP

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
    
    
class FS2023(Base):
    __tablename__ = 'FS2023'

    id = Column(Integer, primary_key=True, autoincrement=True)
    baseDate = Column(Date)
    bizYear = Column(String(50))
    jurir_no = Column(String(50))
    currency = Column(String(10))
    fsCode = Column(String(10))
    fsName = Column(String(100))
    totalAsset2023 = Column(BigInteger)
    totalDebt2023 = Column(BigInteger)
    totalEquity2023 = Column(BigInteger)
    capital2023 = Column(BigInteger)
    revenue2023 = Column(BigInteger)
    operatingIncome2023 = Column(BigInteger)
    earningBeforeTax2023 = Column(BigInteger)
    netIncome2023 = Column(BigInteger)
    debtRatio2023 = Column(Numeric(10, 2))
    margin2023 = Column(Numeric(20, 3))
    turnover2023 = Column(Numeric(20, 3))
    leverage2023 = Column(Numeric(20, 3))
    created_at = Column(TIMESTAMP)