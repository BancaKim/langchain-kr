from datetime import datetime
from pydantic import BaseModel


class GlobalMarketingSchema(BaseModel):

    global_num: int
    corp_code: str
    corp_name: str
    report_nm: str
    rcept_no: str
    rcept_dt: datetime
    created_at: datetime
    html_content: str

    class Config:
        from_attributes = True


class AutocompleteResult(BaseModel):
    result: str
