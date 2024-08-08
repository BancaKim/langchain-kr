from sqlalchemy import Column, Date, Integer, String, ForeignKey, DateTime, Text, func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    email = Column(String(200))
    hashed_password = Column(String(512))
    region_group_name = Column(String(100), nullable=True)
    region_headquarter_name = Column(String(100), nullable=True)
    branch_office_name = Column(String(100), nullable=True)
    position_name = Column(String(100), nullable=True)
    user_rank = Column(String(50), nullable=True)
    region_group_id = Column(Integer, ForeignKey('region_groups.id'))
    region_headquarter_id = Column(Integer, ForeignKey('region_headquarters.id'))
    branch_id = Column(Integer, ForeignKey('branches.id'))
    rank_id = Column(Integer, ForeignKey('ranks.id'))
    position_id = Column(Integer, ForeignKey('positions.id'))

    region_group = relationship("RegionGroup")
    region_headquarter = relationship("RegionHeadquarter")
    branch = relationship("Branch")
    rank = relationship("Rank")
    position = relationship("Position")
    
    notices = relationship("Notice", back_populates="owner")
    qnas = relationship("Qna", back_populates="owner")
    replies = relationship("Reply", back_populates="owner")

class RegionGroup(Base):
    __tablename__ = "region_groups"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)

class RegionHeadquarter(Base):
    __tablename__ = "region_headquarters"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    region_group_id = Column(Integer, ForeignKey('region_groups.id'))

class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    region_headquarter_id = Column(Integer, ForeignKey('region_headquarters.id'))

class Rank(Base):
    __tablename__ = "ranks"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(50), unique=True, index=True)

    positions = relationship("Position", back_populates="rank")

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    rank_id = Column(Integer, ForeignKey('ranks.id'))

    rank = relationship("Rank", back_populates="positions")

class Notice(Base):
    __tablename__ = 'notices'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))
    title = Column(String(100))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="notices")

class Qna(Base):
    __tablename__ = 'qnas'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))
    title = Column(String(100))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="qnas")
    replies = relationship("Reply", back_populates="qna")

class Reply(Base):
    __tablename__ = 'replies'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(100))
    qna_id = Column(Integer, ForeignKey('qnas.id'))
    content = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner = relationship("User", back_populates="replies")
    qna = relationship("Qna", back_populates="replies")


class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)    
    content = Column(Text, nullable=False)    
    username = Column(String(100), nullable=False)
    region_group_name = Column(String(50), nullable=False)
    region_headquarter_name = Column(String(50), nullable=False)
    branch_office_name = Column(String(50), nullable=False)
    position_name = Column(String(50), nullable=False)
    user_rank = Column(String(50), nullable=False)
    corporation_name = Column(String(255), nullable=True)  # New Column
    created_at = Column(DateTime, server_default=func.now())
    
class BusinessCard(Base):
    __tablename__ = "business_cards"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)  # 길이를 255로 지정
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    username = Column(String(100), nullable=False)  # 길이 지정

    # Optional: you can add more fields based on requirements, e.g., company name, phone number, etc.

