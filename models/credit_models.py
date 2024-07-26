from sqlalchemy import Column, DateTime, Integer, Numeric, String, Float, Date, Text
from database import Base


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