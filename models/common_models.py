from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    email = Column(String(200))
    hashed_password = Column(String(512))
    notices = relationship("Notice", back_populates="owner")
    qnas = relationship("Qna", back_populates="owner")
    replies = relationship("Reply", back_populates="owner")

class Notice(Base):
    __tablename__ = 'notices'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))  # 사용자명을 저장할 컬럼 추가
    title = Column(String(100))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="notices")

class Qna(Base):
    __tablename__ = 'qnas'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))  # 사용자명을 저장할 컬럼 추가
    title = Column(String(100))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="qnas")
    replies = relationship("Reply", back_populates="qna")

class Reply(Base):
    __tablename__ = 'replies'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))  # 사용자명을 저장할 컬럼 추가
    qna_id = Column(Integer, ForeignKey('qnas.id'))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="replies")
    qna = relationship("Qna", back_populates="replies")

class Contact(Base):
    __tablename__ = 'contacts'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)  # 길이를 100으로 지정
    email = Column(String(100), index=True)  # 길이를 100으로 지정
    message = Column(Text)
    
class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    file_path = Column(String(255), nullable=True)

    
    
