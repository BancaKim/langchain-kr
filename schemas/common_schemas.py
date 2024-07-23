from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# 회원가입시 데이터 검증
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# 회원로그인시 데이터 검증
class UserLogin(BaseModel):
    username: str
    password: str

# 공지사항 생성 데이터 검증
class NoticeCreate(BaseModel):
    title: str
    content: str

# 공지사항 업데이트 데이터 검증
class NoticeUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

# 공지사항 조회 응답 모델
class NoticeRead(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    user_id: int

    class Config:
        orm_mode: True

# Q&A 생성 데이터 검증
class QnaCreate(BaseModel):
    title: str
    content: str

# Q&A 업데이트 데이터 검증
class QnaUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

# Q&A 조회 응답 모델
class QnaRead(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    user_id: int

    class Config:
        orm_mode: True

# 답글 생성 데이터 검증
class ReplyCreate(BaseModel):
    content: str

# 답글 조회 응답 모델
class ReplyRead(BaseModel):
    id: int
    content: str
    created_at: datetime
    user_id: int
    qna_id: int

    class Config:
        orm_mode: True
    
class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str
