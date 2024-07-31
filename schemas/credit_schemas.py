from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    String,
    Date,
    DECIMAL,
    TIMESTAMP,
    BigInteger,
    ForeignKey,
)


class ReportContentSchema(BaseModel):
    report_num: Optional[int] = None
    corp_code: Optional[str] = None
    corp_name: Optional[str] = None
    report_nm: Optional[str] = None
    rcept_no: Optional[str] = None
    rcept_dt: Optional[datetime] = None
    report_content: Optional[str] = None

    class Config:
        from_attributes = True


class AutocompleteResult(BaseModel):
    result: str
