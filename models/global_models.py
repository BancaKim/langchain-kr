from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base


class GlobalMarketing(Base):
    __tablename__ = "global_marketing"

    global_num = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(24))
    corp_name = Column(String(32))
    report_nm = Column(String(100))
    rcept_no = Column(String(32))
    rcept_dt = Column(DateTime)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    html_content = Column(Text)

    def to_dict(self):
        return {
            "corp_name": self.corp_name,
            "corp_code": self.corp_code,
            "global_num": self.global_num,
            "report_nm": self.report_nm,
            "rcept_no": self.rcept_no,
            "rcept_dt": self.rcept_dt,
            "created_at": self.created_at,
        }
