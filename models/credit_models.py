from sqlalchemy import (
    DateTime,
    Text,
    create_engine,
    Column,
    Integer,
    String,
    Date,
    DECIMAL,
    TIMESTAMP,
    BigInteger,
    Column,
    Integer,
    String,
    Date,
    DECIMAL,
    TIMESTAMP,
    BigInteger,
)
from database import Base
from sqlalchemy.sql import func


# 모델 정의
class ReportContent(Base):
    __tablename__ = "report_content"

    report_num = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(24))
    corp_name = Column(String(32))
    report_nm = Column(String(100))
    rcept_no = Column(String(32))
    rcept_dt = Column(DateTime)
    report_content = Column(Text)

    def to_dict(self):
        return {
            "corp_name": self.corp_name,
            "corp_code": self.corp_code,
            "report_num": self.report_num,
            "report_nm": self.report_nm,
            "rcept_no": self.rcept_no,
            "rcept_dt": self.rcept_dt,
        }


# class CompanyInfo(Base):
#     __tablename__ = "companyInfo"

#     corp_code = Column(String(8), primary_key=True)
#     corp_name = Column(String(255))
#     corp_name_eng = Column(String(255))
#     stock_name = Column(String(255))
#     stock_code = Column(String(6))
#     ceo_nm = Column(String(255))
#     corp_cls = Column(String(1))
#     jurir_no = Column(String(13))
#     bizr_no = Column(String(13))
#     adres = Column(String(255))
#     hm_url = Column(String(255))
#     ir_url = Column(String(255))
#     phn_no = Column(String(20))
#     fax_no = Column(String(20))
#     induty_code = Column(String(10))
#     est_dt = Column(String(8))
#     acc_mt = Column(String(2))


# class FS2023(Base):
#     __tablename__ = "FS2023"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     baseDate = Column(Date)
#     bizYear = Column(Integer)
#     jurir_no = Column(String(50))
#     currency = Column(String(10))
#     fsCode = Column(String(10))
#     fsName = Column(String(100))
#     totalAsset2023 = Column(BigInteger)
#     totalDebt2023 = Column(BigInteger)
#     totalEquity2023 = Column(BigInteger)
#     capital2023 = Column(BigInteger)
#     revenue2023 = Column(BigInteger)
#     operatingIncome2023 = Column(BigInteger)
#     earningBeforeTax2023 = Column(BigInteger)
#     netIncome2023 = Column(BigInteger)
#     debtRatio2023 = Column(DECIMAL(10, 2))
#     margin2023 = Column(DECIMAL(20, 3))
#     turnover2023 = Column(DECIMAL(20, 3))
#     leverage2023 = Column(DECIMAL(20, 3))
#     created_at = Column(TIMESTAMP, server_default=func.now())


# class FS2022(Base):
#     __tablename__ = "FS2022"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     baseDate = Column(Date)
#     bizYear = Column(Integer)
#     jurir_no = Column(String(50))
#     currency = Column(String(10))
#     fsCode = Column(String(10))
#     fsName = Column(String(100))
#     totalAsset2022 = Column(BigInteger)
#     totalDebt2022 = Column(BigInteger)
#     totalEquity2022 = Column(BigInteger)
#     capital2022 = Column(BigInteger)
#     revenue2022 = Column(BigInteger)
#     operatingIncome2022 = Column(BigInteger)
#     earningBeforeTax2022 = Column(BigInteger)
#     netIncome2022 = Column(BigInteger)
#     debtRatio2022 = Column(DECIMAL(10, 2))
#     margin2022 = Column(DECIMAL(20, 3))
#     turnover2022 = Column(DECIMAL(20, 3))
#     leverage2022 = Column(DECIMAL(20, 3))
#     created_at = Column(TIMESTAMP, server_default=func.now())


# class FS2021(Base):
#     __tablename__ = "FS2021"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     baseDate = Column(Date)
#     bizYear = Column(Integer)
#     jurir_no = Column(String(50))
#     currency = Column(String(10))
#     fsCode = Column(String(10))
#     fsName = Column(String(100))
#     totalAsset2021 = Column(BigInteger)
#     totalDebt2021 = Column(BigInteger)
#     totalEquity2021 = Column(BigInteger)
#     capital2021 = Column(BigInteger)
#     revenue2021 = Column(BigInteger)
#     operatingIncome2021 = Column(BigInteger)
#     earningBeforeTax2021 = Column(BigInteger)
#     netIncome2021 = Column(BigInteger)
#     debtRatio2021 = Column(DECIMAL(10, 2))
#     margin2021 = Column(DECIMAL(20, 3))
#     turnover2021 = Column(DECIMAL(20, 3))
#     leverage2021 = Column(DECIMAL(20, 3))
#     created_at = Column(TIMESTAMP, server_default=func.now())
