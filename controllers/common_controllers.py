from datetime import date, datetime
from email.mime.text import MIMEText
import json
import os
import shutil
import smtplib
import logging
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    Query,
    Request,
    Depends,
    HTTPException,
    Form,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from database import SessionLocal
from models.baro_models import CompanyInfo
from models.common_models import (
    Branch,
    BusinessCard,
    Position,
    Post,
    Rank,
    RegionGroup,
    RegionHeadquarter,
    User,
    Notice,
    Qna,
    Reply,
)
from services_def.dependencies import get_db, get_password_hash, verify_password
from schemas.common_schemas import (
    UserCreate,
    NoticeCreate,
    NoticeUpdate,
    QnaCreate,
    QnaUpdate,
    ContactForm,
)
from services_def.email_utils import find_supervisor_email, send_email
from services_def.connection_manager import manager
import urllib.parse

from services_def.news import fetch_naver_news
from services_def.sms import send_sms

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


load_dotenv()  # .env 파일 로드

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# 의존성 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 파일 업로드 관련
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)


def basename(value):
    return os.path.basename(value)


templates.env.filters["basename"] = basename
# 기능 구현 관련


# 테스트
@router.get("/alert")
async def read_root(request: Request):
    return templates.TemplateResponse("alert.html", {"request": request})


# 로그인화면이동
@router.get("/home")
async def read_root(request: Request):
    return templates.TemplateResponse("loginjoin/home.html", {"request": request})


# 회원 가입 페이지
@router.get("/join")
async def read_join(request: Request):
    return templates.TemplateResponse("loginjoin/join.html", {"request": request})


# 회원 가입
@router.post("/signup")
async def signup(signup_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == signup_data.username).first()
    if existing_user:
        return JSONResponse(
            status_code=400,
            content={
                "message": "이미 동일 사용자 이름이 가입되어 있습니다.",
                "message_icon": "error",
            },
        )
    hashed_password = get_password_hash(signup_data.password)

    # 직급 관련
    rank = db.query(Rank).filter(Rank.level == signup_data.rank).first()
    if not rank:
        return JSONResponse(
            status_code=400,
            content={"message": "유효하지 않은 직급입니다.", "message_icon": "error"},
        )

    # 지역그룹, 지역본부, 지점, 직위
    region_group = (
        db.query(RegionGroup).filter(RegionGroup.id == signup_data.region_group).first()
    )
    region_headquarter = (
        db.query(RegionHeadquarter)
        .filter(RegionHeadquarter.id == signup_data.region_headquarter)
        .first()
    )
    branch = db.query(Branch).filter(Branch.id == signup_data.branch).first()
    position = db.query(Position).filter(Position.id == signup_data.position).first()

    new_user = User(
        username=signup_data.username,
        email=signup_data.email,
        hashed_password=hashed_password,
        region_group_id=signup_data.region_group,
        region_headquarter_id=signup_data.region_headquarter,
        branch_id=signup_data.branch,
        rank_id=rank.id,
        position_id=signup_data.position,
        region_group_name=region_group.name if region_group else None,
        region_headquarter_name=region_headquarter.name if region_headquarter else None,
        branch_office_name=branch.name if branch else None,
        user_rank=rank.level,
        position_name=position.name if position else None,
    )
    db.add(new_user)
    try:
        db.commit()
    except Exception as e:
        print(f"Error: {e}")
        logging.info(f"Error2: {e}")
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "message": "회원가입을 실패했습니다. 기입한 내용을 확인해보세요.",
                "message_icon": "error",
            },
        )
    db.refresh(new_user)
    return JSONResponse(
        status_code=200,
        content={
            "message": "회원가입을 성공했습니다.",
            "message_icon": "success",
            "url": "/login",
        },
    )


# 드롭다운 메뉴 클릭하면, 해당되는 부분만 맞춰서 하기
@router.get("/api/region_headquarters/{region_group_id}")
async def get_region_headquarters(region_group_id: int, db: Session = Depends(get_db)):
    region_headquarters = (
        db.query(RegionHeadquarter)
        .filter(RegionHeadquarter.region_group_id == region_group_id)
        .all()
    )
    return [{"id": rh.id, "name": rh.name} for rh in region_headquarters]


@router.get("/api/branches/{region_headquarter_id}")
async def get_branches(region_headquarter_id: int, db: Session = Depends(get_db)):
    branches = (
        db.query(Branch)
        .filter(Branch.region_headquarter_id == region_headquarter_id)
        .all()
    )
    return [{"id": b.id, "name": b.name} for b in branches]


@router.get("/api/positions/{rank_id}")
async def get_positions(rank_id: str, db: Session = Depends(get_db)):
    logging.info(f"Fetching positions for rank_id: {rank_id}")

    rank = db.query(Rank).filter(Rank.level == rank_id).first()
    if not rank:
        logging.warning(f"Rank not found for rank_id: {rank_id}")
        raise HTTPException(status_code=404, detail="Rank not found")

    logging.info(f"Found rank: {rank.id} - {rank.level}")

    positions = rank.positions
    logging.info(f"Found {len(positions)} positions for rank_id: {rank_id}")

    return [{"id": p.id, "name": p.name} for p in positions]


# 로그인
@router.get("/login")
async def login_form(request: Request):
    return templates.TemplateResponse("loginjoin/home.html", {"request": request})


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        request.session["username"] = user.username
        response = RedirectResponse(url="/?login_success=true", status_code=303)
        encoded_username = urllib.parse.quote(request.session["username"])
        response.set_cookie(key="session", value=encoded_username)
        return response
    else:
        return templates.TemplateResponse(
            "loginjoin/home.html",
            {"request": request, "login_failed": True},
        )


# 로그아웃
@router.post("/logout")
async def logout(request: Request):
    request.session.pop("username", None)
    response = RedirectResponse(url="/?logout_success=true", status_code=303)
    response.delete_cookie("session")
    return response


# 공지사항 시작
# 공지사항 목록 조회
@router.get("/notices")
async def list_notices(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    notices = db.query(Notice).all()
    return templates.TemplateResponse(
        "notice/notice.html",
        {"request": request, "notices": notices, "username": username},
    )


# 공지사항 검색
@router.get("/notices/search")
async def search_notices(
    request: Request,
    search_type: str = Query(...),
    search_query: str = Query(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    if search_type == "title":
        notices = db.query(Notice).filter(Notice.title.contains(search_query)).all()
    elif search_type == "content":
        notices = db.query(Notice).filter(Notice.content.contains(search_query)).all()
    elif search_type == "title_content":
        notices = (
            db.query(Notice)
            .filter(
                or_(
                    Notice.title.contains(search_query),
                    Notice.content.contains(search_query),
                )
            )
            .all()
        )
    else:
        notices = db.query(Notice).all()
    return templates.TemplateResponse(
        "notice/notice.html",
        {"request": request, "notices": notices, "username": username},
    )


# 공지사항 생성 페이지
@router.get("/notices/create")
async def create_notice_page(request: Request):
    username = request.session.get("username")
    if username != "admin":
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    return templates.TemplateResponse(
        "notice/notice_create.html", {"request": request, "username": username}
    )


@router.post("/notices/create")
async def create_notice(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    if username != "admin":
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    user = db.query(User).filter(User.username == username).first()

    new_notice = Notice(
        title=title, content=content, user_id=user.id, username=username
    )
    db.add(new_notice)
    db.commit()
    db.refresh(new_notice)
    return RedirectResponse(url="/notices", status_code=303)


# 공지사항 수정 페이지
@router.get("/notices/update/{notice_id}")
async def update_notice_page(
    request: Request, notice_id: int, db: Session = Depends(get_db)
):
    username = request.session.get("username")
    if username != "admin":
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    notice = db.query(Notice).filter(Notice.id == notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
    return templates.TemplateResponse(
        "notice/notice_update.html",
        {"request": request, "notice": notice, "username": username},
    )


# 공지사항 수정
@router.post("/notices/update/{notice_id}")
async def update_notice(
    request: Request,
    notice_id: int,
    title: str = Form(...),
    content: str = Form(...),
    username: str = Form(...),
    db: Session = Depends(get_db),
):
    notice = db.query(Notice).filter(Notice.id == notice_id).first()
    if username != "admin":
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    notice = db.query(Notice).filter(Notice.id == notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
    notice.title = title
    notice.content = content
    notice.username = username
    db.commit()
    db.refresh(notice)
    return RedirectResponse(url="/notices", status_code=303)


# 공지사항 삭제
@router.post("/notices/delete/{notice_id}")
async def delete_notice(
    request: Request, notice_id: int, db: Session = Depends(get_db)
):
    username = request.session.get("username")
    if username != "admin":
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    notice = db.query(Notice).filter(Notice.id == notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
    db.delete(notice)
    db.commit()
    return RedirectResponse(url="/notices", status_code=303)


# 공지사항 상세 조회
@router.get("/notices/{notice_id}")
async def get_notice_detail(
    request: Request, notice_id: int, db: Session = Depends(get_db)
):
    notice = db.query(Notice).filter(Notice.id == notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
    username = request.session.get("username")
    return templates.TemplateResponse(
        "notice/notice_detail.html",
        {"request": request, "notice": notice, "username": username},
    )


# Q&A 시작
# Q&A 목록 조회
@router.get("/qnas")
async def list_qnas(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    qnas = db.query(Qna).all()
    return templates.TemplateResponse(
        "qna/qna.html", {"request": request, "qnas": qnas, "username": username}
    )


# Q&A 검색
@router.get("/qnas/search")
async def search_qnas(
    request: Request,
    search_type: str = Query(...),
    search_query: str = Query(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    if search_type == "title":
        qnas = db.query(Qna).filter(Qna.title.contains(search_query)).all()
    elif search_type == "content":
        qnas = db.query(Qna).filter(Qna.content.contains(search_query)).all()
    elif search_type == "title_content":
        qnas = (
            db.query(Qna)
            .filter(
                or_(
                    Qna.title.contains(search_query), Qna.content.contains(search_query)
                )
            )
            .all()
        )
    else:
        qnas = db.query(Qna).all()
    return templates.TemplateResponse(
        "qna/qna.html", {"request": request, "qnas": qnas, "username": username}
    )


# Q&A 생성 페이지
@router.get("/qnas/create")
async def create_qna_page(request: Request):
    username = request.session.get("username")
    if not username:
        return templates.TemplateResponse(
            "qna/qna_create.html",
            {"request": request, "username": username, "login_required": True},
        )
    return templates.TemplateResponse(
        "qna/qna_create.html", {"request": request, "username": username}
    )


@router.post("/qnas/create")
async def create_qna(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    if not username:
        return templates.TemplateResponse(
            "qna/qna_create.html",
            {"request": request, "username": username, "login_required": True},
        )
    user = db.query(User).filter(User.username == username).first()
    new_qna = Qna(title=title, content=content, user_id=user.id, username=user.username)
    db.add(new_qna)
    db.commit()
    db.refresh(new_qna)
    return RedirectResponse(url="/qnas", status_code=303)


# Q&A 수정 페이지
@router.get("/qnas/update/{qna_id}")
async def update_qna_page(request: Request, qna_id: int, db: Session = Depends(get_db)):
    qna = db.query(Qna).filter(Qna.id == qna_id).first()
    username = request.session.get("username")
    if not qna:
        raise HTTPException(status_code=404, detail="Q&A를 찾을 수 없습니다.")
    return templates.TemplateResponse(
        "qna/qna_update.html", {"request": request, "qna": qna, "username": username}
    )


# Q&A 수정
@router.post("/qnas/update/{qna_id}")
async def update_qna(
    request: Request,
    qna_id: int,
    title: str = Form(...),
    content: str = Form(...),
    username: str = Form(...),
    db: Session = Depends(get_db),
):
    qna = db.query(Qna).filter(Qna.id == qna_id).first()
    if not qna:
        raise HTTPException(status_code=404, detail="Q&A를 찾을 수 없습니다.")
    qna.title = title
    qna.content = content
    qna.username = username
    db.commit()
    db.refresh(qna)
    return RedirectResponse(url="/qnas", status_code=303)


# Q&A 삭제
@router.post("/qnas/delete/{qna_id}")
async def delete_qna(request: Request, qna_id: int, db: Session = Depends(get_db)):
    qna = db.query(Qna).filter(Qna.id == qna_id).first()
    if not qna:
        raise HTTPException(status_code=404, detail="Q&A를 찾을 수 없습니다.")
    db.delete(qna)
    db.commit()
    return RedirectResponse(url="/qnas", status_code=303)


# Q&A 상세 조회
@router.get("/qnas/{qna_id}")
async def qna_detail(qna_id: int, request: Request, db: Session = Depends(get_db)):
    qna = db.query(Qna).filter(Qna.id == qna_id).first()
    if not qna:
        raise HTTPException(status_code=404, detail="Q&A를 찾을 수 없습니다.")
    replies = db.query(Reply).filter(Reply.qna_id == qna_id).all()
    username = request.session.get("username")
    return templates.TemplateResponse(
        "qna/qna_detail.html",
        {"request": request, "qna": qna, "replies": replies, "username": username},
    )


# Q&A 답글 달기
@router.post("/qnas/{qna_id}/reply")
async def create_reply(
    qna_id: int,
    request: Request,
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    user = db.query(User).filter(User.username == username).first()
    new_reply = Reply(
        content=content, qna_id=qna_id, user_id=user.id, username=username
    )
    db.add(new_reply)
    db.commit()
    db.refresh(new_reply)
    return RedirectResponse(url=f"/qnas/{qna_id}", status_code=303)


# # 섭외등록 이메일 보내기 페이지
# @router.get("/contact2")
# async def read_contact(request: Request):
#     username = request.session.get("username")
#     return templates.TemplateResponse(
#         "contact/contact2.html", {"request": request, "username": username}
#     )


# # 섭외등록 이메일 보내기
# @router.post("/contact2")
# async def submit_contact_form(
#     request: Request,
#     background_tasks: BackgroundTasks,
#     name: str = Form(...),
#     email: str = Form(...),
#     message: str = Form(...),

#     db: Session = Depends(get_db),
# ):
#     send_email(
#         background_tasks,
#         "섭외등록 내용이 도착했습니다",
#         "sjung8009@naver.com",
#         f"업체(키맨) 이름: {name}\n업체(키맨) 이메일: {email}\n섭외 메모: {message}",
#     )
#     return templates.TemplateResponse(
#         "contact/contact2.html",
#         {"request": request, "message": "Contact form submitted successfully"},
#     )


# 섭외등록 시작
# 섭외등록 목록 조회
@router.get("/contact", response_class=HTMLResponse)
async def get_posts(
    request: Request, db: Session = Depends(get_db), page: int = Query(1, alias="page")
):
    posts_per_page = 10
    offset = (page - 1) * posts_per_page
    total_posts = db.query(Post).count()
    posts = db.query(Post).offset(offset).limit(posts_per_page).all()
    total_pages = (total_posts + posts_per_page - 1) // posts_per_page
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact.html",
        {
            "request": request,
            "posts": posts,
            "username": username,
            "page": page,
            "total_pages": total_pages,
        },
    )


# 섭외등록 생성
@router.post("/contact/create")
async def create_post(
    request: Request,
    background_tasks: BackgroundTasks,
    content: str = Form(None),
    corporation_name: str = Form(...),  # Receiving corporation name
    contact_type: str = Form(...),
    contact_method: str = Form(...),
    send_email_flag: str = Form(None),
    send_sms_flag: str = Form(None),  # 새로 추가된 SMS 전송 플래그
    db: Session = Depends(get_db)
):
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")

    # Create the new post
    new_post = Post(
        content=content,
        username=user.username,
        region_group_name=user.region_group.name,
        region_headquarter_name=user.region_headquarter.name,
        branch_office_name=user.branch.name,
        position_name=user.position.name,
        user_rank=user.rank.level,
        corporation_name=corporation_name,
        contact_type=contact_type,
        contact_method=contact_method,
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    # # Remove corporation name from session if it exists
    # if 'corporation_name' in request.session:
    #     del request.session['corporation_name']

    # 이메일 전송 로직
    if send_email_flag == "true":  # send_email_flag가 'true'일 때만 이메일 전송
        supervisor_email = find_supervisor_email(db, user.region_headquarter.name)
        if supervisor_email:
            email_content = {
                "username": username,
                "content": content,
                "corporation_name": corporation_name,
            }
            send_email(
                background_tasks,
                "섭외등록 내용이 도착했습니다",
                supervisor_email,
                "contact/email_template.html",
                email_content,
            )
            
    # 문자 전송 로직
    if send_sms_flag == 'true':
        background_tasks.add_task(send_sms, username=username, corporation_name=corporation_name, content=content)

    return RedirectResponse(url="/contact", status_code=303)
    


# 섭외등록 생성 페이지
@router.get("/contact/create")
async def create_post_page(request: Request):
    username = request.session.get("username")
    # Fetch company_info from the session or database
    corporation_name = request.session.get("corporation_name", None)
    company_info = request.session.get("company_info", None)
    login_required = not bool(username)  # 로그인 여부 확인
    return templates.TemplateResponse(
        "contact/contact_create.html",
        {
            "request": request,
            "username": username,
            "corporation_name": corporation_name,
            "company_info": company_info,
            "login_required": login_required,
        },
    )


# 검색 및 페이지네이션 처리 함수
@router.get("/contact/search")
async def search_contacts(
    request: Request,
    search_type: str,
    search_query: str,
    page: int = 1,
    db: Session = Depends(get_db),
):
    page_size = 10
    offset = (page - 1) * page_size

    if search_type == "title":
        posts = (
            db.query(Post)
            .filter(Post.title.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "content":
        posts = (
            db.query(Post)
            .filter(Post.content.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "title_content":
        posts = (
            db.query(Post)
            .filter(
                Post.title.contains(search_query) | Post.content.contains(search_query)
            )
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "region_group":
        posts = (
            db.query(Post)
            .filter(Post.region_group_name.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "region_headquarter":
        posts = (
            db.query(Post)
            .filter(Post.region_headquarter_name.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "branch_office":
        posts = (
            db.query(Post)
            .filter(Post.branch_office_name.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    elif search_type == "corporation_name":
        posts = (
            db.query(Post)
            .filter(Post.corporation_name.contains(search_query))
            .offset(offset)
            .limit(page_size)
            .all()
        )
    else:
        posts = db.query(Post).offset(offset).limit(page_size).all()

    total_posts = (
        db.query(Post)
        .filter(
            (Post.title.contains(search_query))
            | (Post.content.contains(search_query))
            | (Post.region_group_name.contains(search_query))
            | (Post.region_headquarter_name.contains(search_query))
            | (Post.branch_office_name.contains(search_query))
            | (Post.corporation_name.contains(search_query))
        )
        .count()
    )
    total_pages = (total_posts // page_size) + (1 if total_posts % page_size > 0 else 0)

    return templates.TemplateResponse(
        "contact/contact.html",
        {
            "request": request,
            "posts": posts,
            "search_type": search_type,
            "search_query": search_query,
            "page": page,
            "total_pages": total_pages,
        },
    )


# 섭외등록 상세 조회
@router.get("/contact/{post_id}", response_class=HTMLResponse)
async def read_post(request: Request, post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        return RedirectResponse(url="/contact", status_code=303)

    username = request.session.get("username")

    region_group_name = post.region_group_name
    region_headquarter_name = post.region_headquarter_name
    branch_office_name = post.branch_office_name
    position_name = post.position_name
    user_rank = post.user_rank

    return templates.TemplateResponse(
        "contact/contact_detail.html",
        {
            "request": request,
            "post": post,
            "username": username,
            "region_group_name": region_group_name,
            "region_headquarter_name": region_headquarter_name,
            "branch_office_name": branch_office_name,
            "position_name": position_name,
            "user_rank": user_rank,
        },
    )


# 섭외등록 수정 페이지
@router.get("/contact/update/{post_id}")
async def update_post_page(
    request: Request, post_id: int, db: Session = Depends(get_db)
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        return RedirectResponse(url="/contact", status_code=303)

    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact_update.html",
        {"request": request, "post": post, "username": username},
    )


# 섭외등록 수정
@router.post("/contact/update/{post_id}")
async def update_post(
    request: Request,
    post_id: int,
    title: str = Form(...),
    content: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        return RedirectResponse(url="/contact", status_code=303)

    file_path = post.file_path
    if file and file.filename:
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail="파일 업로드에 실패했습니다.")
        file_path = f"/{file_path}"  # Static URL for accessing the file

    post.title = title
    post.content = content
    post.file_path = file_path

    db.commit()
    db.refresh(post)
    return RedirectResponse(url=f"/contact/{post.id}", status_code=303)


# 섭외등록 삭제
@router.post("/contact/delete/{post_id}")
async def delete_post(request: Request, post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        return RedirectResponse(url="/contact", status_code=303)

    db.delete(post)
    db.commit()
    return RedirectResponse(url="/contact", status_code=303)


# 파일 다운로드 엔드포인트
@router.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# 지도기능 - 주소
# 카카오 지도 API
@router.get("/search", response_class=HTMLResponse)
async def get_search_page(request: Request):
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    return templates.TemplateResponse(
        "contact/map.html", {"request": request, "kakao_map_api_key": kakao_map_api_key}
    )


@router.post("/search", response_class=HTMLResponse)
async def search_location(request: Request):
    return templates.TemplateResponse("contact/map.html", {"request": request})


# 지도기능 - 키워드
# 카카오 지도 API
@router.get("/search2", response_class=HTMLResponse)
async def get_search_page(request: Request):
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    return templates.TemplateResponse(
        "contact/map2.html",
        {"request": request, "kakao_map_api_key": kakao_map_api_key},
    )


@router.post("/search2", response_class=HTMLResponse)
async def search_location(request: Request):
    return templates.TemplateResponse("contact/map2.html", {"request": request})


@router.get("/contact3")
async def get_chat_page(request: Request):
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact3.html", {"request": request, "username": username}
    )


@router.websocket("/ws/chat/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(websocket, username)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            if message_data.get("type") == "whisper":
                target_username = message_data.get("target")
                content = message_data.get("content")
                await manager.send_personal_message(
                    json.dumps(
                        {"sender": username, "content": content, "type": "whisper"}
                    ),
                    target_username,
                )
            else:
                await manager.broadcast(
                    json.dumps(
                        {"sender": username, "content": message_data.get("content")}
                    )
                )
    except WebSocketDisconnect:
        manager.disconnect(username)
        await manager.broadcast(
            json.dumps(
                {
                    "sender": "System",
                    "content": f"{username} 사용자가 채팅에서 퇴장하였습니다.",
                }
            )
        )


# 기능홈페이지(임시)
@router.get("/contact4")
async def read_contact(request: Request):
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact4.html", {"request": request, "username": username}
    )


# 기능홈페이지
@router.get("/contact55")
async def read_contact(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    posts = db.query(
        Post.corporation_name,
        Post.content,
        Post.username,
        Post.created_at,
        Post.contact_type,
        Post.contact_method,
    ).all()

    return templates.TemplateResponse(
        "contact/contact55.html",
        {"request": request, "username": username, "posts": posts},
    )


@router.get("/contact55/search")
async def search_contacts(
    request: Request,
    search_query: str = Query(None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    per_page = 10
    query = db.query(Post)

    if search_query:
        query = query.filter(
            or_(
                Post.content.contains(search_query),
                Post.username.contains(search_query),
                Post.corporation_name.contains(search_query),
            )
        )

    total_posts = query.count()
    total_pages = (total_posts - 1) // per_page + 1

    posts = query.offset((page - 1) * per_page).limit(per_page).all()

    return templates.TemplateResponse(
        "contact/contact55.html",
        {
            "request": request,
            "posts": posts,
            "page": page,
            "total_pages": total_pages,
            "search_query": search_query,
            "username": username
        }

    )


# 세부정보
@router.get("/contact5")
async def read_contact(
    request: Request, jurir_no: str = Query(...), db: Session = Depends(get_db)
):
    username = request.session.get("username")
    company_info = (
        db.query(CompanyInfo).filter(CompanyInfo.jurir_no == jurir_no).first()
    )
    if not company_info:
        raise HTTPException(status_code=404, detail="Company not found")

    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    if not kakao_map_api_key:
        raise HTTPException(status_code=500, detail="Kakao Map API key is not set")

    # 명함 데이터를 회사 이름으로 필터링하여 가져오기
    business_cards = (
        db.query(BusinessCard)
        .filter(BusinessCard.corporation_name == company_info.corp_name)
        .all()
    )

    # # 명함 데이터를 가져오기
    # business_cards = db.query(BusinessCard).all()

    # 포스트 데이터를 가져오기
    # posts = db.query(Post).all()

    # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
    posts = db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()

    # 뉴스 기사 가져오기
    try:
        news_articles = fetch_naver_news(company_info.corp_name)
    except HTTPException as e:
        news_error = str(e)

    # logging.info(f"Address: {company_info.adres}")  # 디버깅을 위해 주소 출력
    # logging.info(f"API Key: {kakao_map_api_key}")  # API 키 확인
    # logging.info(f"Username: {username}")  # 사용자명 확인

    return templates.TemplateResponse(
        "contact/contact5.html",
        {
            "request": request,
            "username": username,
            "company_info": company_info,
            "kakao_map_api_key": kakao_map_api_key,
            "news": news_articles,
            "adres": company_info.adres,
            "business_cards": business_cards,  # (필터링된) 명함 데이터를 템플릿으로 전달
            "posts": posts,  # (필터링된) 포스트 데이터를 템플릿으로 전달
        },
    )


@router.post("/contact5")
async def show_company_details(
    request: Request,
    db: Session = Depends(get_db),
    name: Optional[str] = Form(None),
    search_type: Optional[str] = Form(None),
):
    username = request.session.get("username")
    try:
        company_info = None
        news_articles = []
        news_error = None

        if name:
            if search_type == "company_name":
                company_info = (
                    db.query(CompanyInfo)
                    .filter(func.trim(CompanyInfo.corp_name) == name)
                    .first()
                )
            elif search_type == "company_code":
                company_info = (
                    db.query(CompanyInfo)
                    .filter(func.trim(CompanyInfo.corp_code) == name)
                    .first()
                )

        if not company_info:
            raise HTTPException(status_code=404, detail="Company not found")

        # 뉴스 기사 가져오기
        try:
            news_articles = fetch_naver_news(company_info.corp_name)
        except HTTPException as e:
            news_error = str(e)

        kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
        if not kakao_map_api_key:
            raise HTTPException(status_code=500, detail="Kakao Map API key is not set")

        # 명함 데이터를 가져오기
        # business_cards = db.query(BusinessCard).all()
        business_cards = (
            db.query(BusinessCard)
            .filter(BusinessCard.corporation_name == company_info.corp_name)
            .all()
        )

        # 포스트 데이터를 가져오기
        # posts = db.query(Post).all()

        # 포스트 데이터를 회사 이름으로 필터링하여 가져오기
        posts = (
            db.query(Post).filter(Post.corporation_name == company_info.corp_name).all()
        )

        return templates.TemplateResponse(
            "contact/contact5.html",
            {
                "request": request,
                "username": username,
                "company_info": company_info,
                "news": news_articles,
                "corporation_name": company_info.corp_name,
                "error": news_error if news_error else None,
                "kakao_map_api_key": kakao_map_api_key,
                "adres": company_info.adres,
                "business_cards": business_cards,  # 명함 데이터를 템플릿으로 전달
                "posts": posts,  # 포스트 데이터를 템플릿으로 전달
            },
        )
    except Exception as e:
        logging.error("An error occurred:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 비밀번호 가져오기, 채팅방 비밀번호
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")


class PasswordVerification(BaseModel):
    password: str


@router.post("/verify_password2")
async def verify_password2(data: PasswordVerification):
    correct_password = CHAT_PASSWORD

    logging.info(f"Received password verification request. Password: {data.password}")
    # logger = logging.getLogger(data.password)
    # logger.debug(data.password)

    result = data.password == correct_password
    logging.info(f"Verifying password. Result: {result}")
    return {"success": result}


@router.get("/news", response_class=HTMLResponse)
async def show_news_page(request: Request):
    return templates.TemplateResponse("contact/news.html", {"request": request})


@router.post("/news", response_class=HTMLResponse)
async def search_news(request: Request, corporation_name: str = Form(...)):
    try:
        news_articles = fetch_naver_news(corporation_name)
        return templates.TemplateResponse(
            "contact/news.html",
            {
                "request": request,
                "news": news_articles,
                "corporation_name": corporation_name,
            },
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "contact/news.html",
            {
                "request": request,
                "error": str(e),
                "news": [],
                "corporation_name": corporation_name,
            },
        )


@router.post("/upload-business-card")
async def upload_business_card(
    username: str = Form(...),
    file: UploadFile = File(...),
    corporation_name: str = Form(...),  # 회사명 필드 추가
    db: Session = Depends(get_db),
):
    try:
        # 파일 저장
        file_location = f"static/business_cards/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 데이터베이스에 파일 정보 저장
        business_card = BusinessCard(
            filename=file.filename,
            username=username,
            corporation_name=corporation_name,  # 회사명 필드 저장
            created_at=datetime.utcnow(),  # 현재 시간을 UTC로 저장
        )
        db.add(business_card)
        db.commit()

        # 명함 목록 페이지로 리디렉션
        return RedirectResponse(
            url=f"/card?corporation_name={corporation_name}", status_code=303
        )
    except Exception as e:
        db.rollback()  # 오류 발생 시 데이터베이스 롤백
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/card")
async def show_cards(
    request: Request, corporation_name: str = Query(...), db: Session = Depends(get_db)
):
    username = request.session.get("username")
    # 명함 데이터를 회사 이름으로 필터링하여 가져오기

    cards = db.query(BusinessCard).filter(BusinessCard.corporation_name == corporation_name).all()
    
    return templates.TemplateResponse("contact/card.html", {"request": request, "cards": cards, "username": username, "corporation_name": corporation_name})

    cards = (
        db.query(BusinessCard)
        .filter(BusinessCard.corporation_name == corporation_name)
        .all()
    )

    return templates.TemplateResponse(
        "contact/card.html",
        {
            "request": request,
            "cards": cards,
            "username": username,
            "corporation_name": corporation_name,
        },
    )

