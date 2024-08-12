# app/models/data_model.py
from pydantic import BaseModel


class DataModel(BaseModel):
    loan: float
    IR: float
    asset2023: float
    debt2023: float
    equity2023: float
    revenue2023: float
    operatingincome2023: float
    EBT2023: float
    margin2023: float
    turnover2023: float
    leverage2023: float
    asset2022: float
    debt2022: float
    equity2022: float
    revenue2022: float
    operatingincome2022: float
    EBT2022: float
    margin2022: float
    turnover2022: float
    leverage2022: float
    asset2021: float
    debt2021: float
    equity2021: float
    revenue2021: float
    operatingincome2021: float
    EBT2021: float
    margin2021: float
    turnover2021: float
    leverage2021: float