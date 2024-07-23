from pydantic import BaseModel
from typing import Optional

class CompanyInfoSchema(BaseModel):
    corp_code: str
    corp_name: Optional[str] = None
    corp_name_eng: Optional[str] = None
    stock_name: Optional[str] = None
    stock_code: Optional[str] = None
    ceo_nm: Optional[str] = None
    corp_cls: Optional[str] = None
    jurir_no: Optional[str] = None
    bizr_no: Optional[str] = None
    adres: Optional[str] = None
    hm_url: Optional[str] = None
    ir_url: Optional[str] = None
    phn_no: Optional[str] = None
    fax_no: Optional[str] = None
    induty_code: Optional[str] = None
    est_dt: Optional[str] = None
    acc_mt: Optional[str] = None

    class Config:
        from_attributes = True