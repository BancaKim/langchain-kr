from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, Numeric, String, Float, Date, BigInteger, TIMESTAMP, Text

from database import Base

## dart 기준 회사 정보
class CompanyInfoFS(Base):
    __tablename__ = 'companyInfoFS'
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
    
class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10))
    corp_code = Column(String(10))
    corp_name = Column(String(255))
    listing_date = Column(Date)
    latest_date = Column(Date)
    latest_price = Column(Float)
    cagr_since_listing = Column(Float)
    vol_since_listing = Column(Float)
    cagr_1y = Column(Float)
    vol_1y = Column(Float)
    cagr_3y = Column(Float)
    vol_3y = Column(Float)
    cagr_5y = Column(Float)
    vol_5y = Column(Float)
    stock_count = Column(Integer)
    per_value = Column(String(10))
    pbr_value = Column(String(10))
    market_capitalization = Column(Float)
    timestamp = Column(TIMESTAMP)
    reference = Column(String(255))
    
# class FS2022(Base):
#     __tablename__ = 'FS2022'

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     baseDate = Column(Date)
#     bizYear = Column(String(50))
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
#     debtRatio2022 = Column(Numeric(10, 2))
#     margin2022 = Column(Numeric(20, 3))
#     turnover2022 = Column(Numeric(20, 3))
#     leverage2022 = Column(Numeric(20, 3))
#     created_at = Column(TIMESTAMP)
    
# class FS2023(Base):
#     __tablename__ = 'FS2023'

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     baseDate = Column(Date)
#     bizYear = Column(String(50))
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
#     debtRatio2023 = Column(Numeric(10, 2))
#     margin2023 = Column(Numeric(20, 3))
#     turnover2023 = Column(Numeric(20, 3))
#     leverage2023 = Column(Numeric(20, 3))
#     created_at = Column(TIMESTAMP)

class FS2023(Base):
    __tablename__ = 'FS2023'

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    baseDate = Column(Date)  # 기준 날짜
    bizYear = Column(String(50))  # 사업 연도
    jurir_no = Column(String(50))  # 법인 등록 번호
    currency = Column(String(10))  # 통화
    fsCode = Column(String(10))  # 재무제표 코드
    fsName = Column(String(100))  # 재무제표 이름
    totalAsset2023 = Column(BigInteger)  # 자산 총계
    totalDebt2023 = Column(BigInteger)  # 부채 총계
    totalEquity2023 = Column(BigInteger)  # 자본 총계
    capital2023 = Column(BigInteger)  # 자본금
    revenue2023 = Column(BigInteger)  # 매출액
    operatingIncome2023 = Column(BigInteger)  # 영업이익
    earningBeforeTax2023 = Column(BigInteger)  # 법인세 차감전 순이익
    netIncome2023 = Column(BigInteger)  # 당기순이익
    debtRatio2023 = Column(Numeric(10, 2))  # 부채비율
    margin2023 = Column(Numeric(20, 3))  # 이익률
    turnover2023 = Column(Numeric(20, 3))  # 회전율
    leverage2023 = Column(Numeric(20, 3))  # 레버리지
    created_at = Column(TIMESTAMP)  # 생성 시간
    FS_url = Column(String(100))  # 재무제표 URL
    FS_detail = Column(Text)  # 재무제표 상세 내용
    
class FS2022(Base):
    __tablename__ = 'FS2022'

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    baseDate = Column(Date)  # 기준 날짜
    bizYear = Column(String(50))  # 사업 연도
    jurir_no = Column(String(50))  # 법인 등록 번호
    currency = Column(String(10))  # 통화
    fsCode = Column(String(10))  # 재무제표 코드
    fsName = Column(String(100))  # 재무제표 이름
    totalAsset2022 = Column(BigInteger)  # 자산 총계
    totalDebt2022 = Column(BigInteger)  # 부채 총계
    totalEquity2022 = Column(BigInteger)  # 자본 총계
    capital2022 = Column(BigInteger)  # 자본금
    revenue2022 = Column(BigInteger)  # 매출액
    operatingIncome2022 = Column(BigInteger)  # 영업이익
    earningBeforeTax2022 = Column(BigInteger)  # 법인세 차감전 순이익
    netIncome2022 = Column(BigInteger)  # 당기순이익
    debtRatio2022 = Column(Numeric(10, 2))  # 부채비율
    margin2022 = Column(Numeric(20, 3))  # 이익률
    turnover2022 = Column(Numeric(20, 3))  # 회전율
    leverage2022 = Column(Numeric(20, 3))  # 레버리지
    created_at = Column(TIMESTAMP)  # 생성 시간
    FS_url = Column(String(100))  # 재무제표 URL
    FS_detail = Column(Text)  # 재무제표 상세 내용
    
class FS2021(Base):
    __tablename__ = 'FS2021'

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    baseDate = Column(Date)  # 기준 날짜
    bizYear = Column(String(50))  # 사업 연도
    jurir_no = Column(String(50))  # 법인 등록 번호
    currency = Column(String(10))  # 통화
    fsCode = Column(String(10))  # 재무제표 코드
    fsName = Column(String(100))  # 재무제표 이름
    totalAsset2021 = Column(BigInteger)  # 자산 총계
    totalDebt2021 = Column(BigInteger)  # 부채 총계
    totalEquity2021 = Column(BigInteger)  # 자본 총계
    capital2021 = Column(BigInteger)  # 자본금
    revenue2021 = Column(BigInteger)  # 매출액
    operatingIncome2021 = Column(BigInteger)  # 영업이익
    earningBeforeTax2021 = Column(BigInteger)  # 법인세 차감전 순이익
    netIncome2021 = Column(BigInteger)  # 당기순이익
    debtRatio2021 = Column(Numeric(10, 2))  # 부채비율
    margin2021 = Column(Numeric(20, 3))  # 이익률
    turnover2021 = Column(Numeric(20, 3))  # 회전율
    leverage2021 = Column(Numeric(20, 3))  # 레버리지
    created_at = Column(TIMESTAMP)  # 생성 시간
    FS_url = Column(String(100))  # 재무제표 URL
    FS_detail = Column(Text)  # 재무제표 상세 내용
    
class FS2020(Base):
    __tablename__ = 'FS2020'

    id = Column(Integer, primary_key=True, autoincrement=True)  # 기본 키
    baseDate = Column(Date)  # 기준 날짜
    bizYear = Column(String(50))  # 사업 연도
    jurir_no = Column(String(50))  # 법인 등록 번호
    currency = Column(String(10))  # 통화
    fsCode = Column(String(10))  # 재무제표 코드
    fsName = Column(String(100))  # 재무제표 이름
    totalAsset2020 = Column(BigInteger)  # 자산 총계
    totalDebt2020 = Column(BigInteger)  # 부채 총계
    totalEquity2020 = Column(BigInteger)  # 자본 총계
    capital2020 = Column(BigInteger)  # 자본금
    revenue2020 = Column(BigInteger)  # 매출액
    operatingIncome2020 = Column(BigInteger)  # 영업이익
    earningBeforeTax2020 = Column(BigInteger)  # 법인세 차감전 순이익
    netIncome2020 = Column(BigInteger)  # 당기순이익
    debtRatio2020 = Column(Numeric(10, 2))  # 부채비율
    margin2020 = Column(Numeric(20, 3))  # 이익률
    turnover2020 = Column(Numeric(20, 3))  # 회전율
    leverage2020 = Column(Numeric(20, 3))  # 레버리지
    created_at = Column(TIMESTAMP)  # 생성 시간
    FS_url = Column(String(100))  # 재무제표 URL
    FS_detail = Column(Text)  # 재무제표 상세 내용



class Favorite(Base):
    __tablename__ = 'favorites'
    id = Column(Integer, primary_key=True, autoincrement=True)   
    username = Column(String(100))
    corp_code = Column(String(8))
    
    
class RecentView(Base):
    __tablename__ = 'recent_views'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100))
    corp_code = Column(String(8))
    created_at = Column(DateTime, default=datetime.utcnow)

