from email.mime.text import MIMEText
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
from sqlalchemy import or_
from sqlalchemy.orm import Session
from models.common_models import Branch, Contact, Position, Post, Rank, RegionGroup, RegionHeadquarter, User, Notice, Qna, Reply
from services_def.dependencies import get_db, get_password_hash, verify_password
from schemas.common_schemas import (
    UserCreate,
    NoticeCreate,
    NoticeUpdate,
    QnaCreate,
    QnaUpdate,
    ContactForm,
)
from services_def.email_utils import send_email
from services_def.connection_manager import manager
import urllib.parse

# logger = logging.getLogger('uvicorn.error')
# logger.setLevel(logging.DEBUG)


load_dotenv()  # .env 파일 로드

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# 파일 업로드 관련
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)


def basename(value):
    return os.path.basename(value)


templates.env.filters["basename"] = basename

# 로그인화면이동


@router.get('/home')
async def read_root(request: Request):
    return templates.TemplateResponse('loginjoin/home.html', {"request": request})

# 회원 가입 페이지


@router.get("/join")
async def read_join(request: Request):
    return templates.TemplateResponse("loginjoin/join.html", {"request": request})


# 회원 가입
@router.post("/signup")
async def signup(signup_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(
        User.username == signup_data.username).first()
    if existing_user:
        return JSONResponse(
            status_code=400, content={"message": "이미 동일 사용자 이름이 가입되어 있습니다.", "message_icon": "error"}
        )
    hashed_password = get_password_hash(signup_data.password)

    # Convert rank to its corresponding ID if necessary
    rank = db.query(Rank).filter(Rank.level == signup_data.rank).first()
    if not rank:
        return JSONResponse(
            status_code=400, content={"message": "유효하지 않은 직급입니다.", "message_icon": "error"}
        )

    # Fetch related data names
    region_group = db.query(RegionGroup).filter(
        RegionGroup.id == signup_data.region_group).first()
    region_headquarter = db.query(RegionHeadquarter).filter(
        RegionHeadquarter.id == signup_data.region_headquarter).first()
    branch = db.query(Branch).filter(Branch.id == signup_data.branch).first()
    position = db.query(Position).filter(
        Position.id == signup_data.position).first()

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
        position_name=position.name if position else None
    )
    db.add(new_user)
    try:
        db.commit()
    except Exception as e:
        print(f"Error: {e}")
        logging.info(f"Error2: {e}")
        db.rollback()
        return JSONResponse(
            status_code=500, content={"message": "회원가입이 실패했습니다. 기입한 내용을 확인해보세요.", "message_icon": "error"}
        )
    db.refresh(new_user)
    return JSONResponse(
        status_code=200, content={"message": "회원가입이 성공했습니다.", "message_icon": "success", "url": "/login"}
    )


# API for loading dynamic data
@router.get("/api/region_headquarters/{region_group_id}")
async def get_region_headquarters(region_group_id: int, db: Session = Depends(get_db)):
    region_headquarters = db.query(RegionHeadquarter).filter(
        RegionHeadquarter.region_group_id == region_group_id).all()
    return [{"id": rh.id, "name": rh.name} for rh in region_headquarters]


@router.get("/api/branches/{region_headquarter_id}")
async def get_branches(region_headquarter_id: int, db: Session = Depends(get_db)):
    branches = db.query(Branch).filter(
        Branch.region_headquarter_id == region_headquarter_id).all()
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
        response = templates.TemplateResponse(
            "loginjoin/home.html",
            {"request": request, "message": "로그인이 성공했습니다.",
                "message_icon": "success", "url": "/"},
        )
        encoded_username = urllib.parse.quote(
            request.session["username"])  # URL 인코딩
        response.set_cookie(key="session", value=encoded_username)
        return response
    else:
        response = templates.TemplateResponse(
            "loginjoin/home.html",
            {"request": request, "message": "로그인이 실패했습니다.", "url": "home"},
        )
        return response


# 로그아웃
@router.post("/logout")
async def logout(request: Request):
    request.session.pop("username", None)
    response = templates.TemplateResponse(
        "loginjoin/home.html",
        {"request": request, "message": "로그아웃되었습니다.",
            "message_icon": "success", "url": "/"},
    )
    response.delete_cookie("session")
    return response


# 공지사항 목록 조회
@router.get("/notices")
async def list_notices(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    notices = db.query(Notice).all()
    return templates.TemplateResponse(
        "notice/notice.html", {"request": request,
                               "notices": notices, "username": username}
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
        notices = db.query(Notice).filter(
            Notice.title.contains(search_query)).all()
    elif search_type == "content":
        notices = db.query(Notice).filter(
            Notice.content.contains(search_query)).all()
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
        "notice/notice.html", {"request": request,
                               "notices": notices, "username": username}
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

    # 인위적으로 지연 추가 (예: 5초)
    time.sleep(5)

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


# Q&A 목록 조회
@router.get("/qnas")
async def list_qnas(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    qnas = db.query(Qna).all()
    return templates.TemplateResponse(
        "qna/qna.html", {"request": request,
                        "qnas": qnas, "username": username}
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
                    Qna.title.contains(
                        search_query), Qna.content.contains(search_query)
                )
            )
            .all()
        )
    else:
        qnas = db.query(Qna).all()
    return templates.TemplateResponse(
        "qna/qna.html", {"request": request,
                         "qnas": qnas, "username": username}
    )


# Q&A 생성 페이지
@router.get("/qnas/create")
async def create_qna_page(request: Request):
    username = request.session.get("username")
    if not username:
        return templates.TemplateResponse(
            "qna/qna_create.html", {"request": request,
                                    "username": username, "login_required": True}
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
            "qna/qna_create.html", {"request": request,
                                    "username": username, "login_required": True}
        )
    user = db.query(User).filter(User.username == username).first()
    new_qna = Qna(title=title, content=content,
                  user_id=user.id, username=user.username)
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
        "qna/qna_update.html", {"request": request,
                                "qna": qna, "username": username}
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


# 파일첨부 게시판형식 (섭외등록)
# 게시글 목록 조회
@router.get("/contact")
async def get_posts(request: Request, db: Session = Depends(get_db)):
    posts = db.query(Post).all()
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact.html", {"request": request,
                                "posts": posts, "username": username}
    )
    
# 섭외등록 생성
@router.post("/contact/create")
async def create_post(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    content: str = Form(...),
    corporation_name: str = Form(...),
    file: UploadFile = File(None),
    send_email_flag: str = Form(None),
    db: Session = Depends(get_db)    
):
    file_path = None
    if file and file.filename:
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail="파일 업로드에 실패했습니다.")

    username = request.session.get("username")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=400, detail="사용자를 찾을 수 없습니다.")
    file_path = f"/{file_path}"  # Static URL for accessing the file

    new_post = Post(
        title=title,
        content=content,
        file_path=file_path,
        username=user.username,
        region_group_name=user.region_group.name,
        region_headquarter_name=user.region_headquarter.name,
        branch_office_name=user.branch.name,
        position_name=user.position.name,
        user_rank=user.rank.level,
        corporation_name=corporation_name  # 세션 또는 사용자가 입력한 법인명
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    # 세션에서 법인명 제거
    if 'corporation_name' in request.session:
        del request.session['corporation_name']

    # 이메일 전송 로직
    if send_email_flag:
        email_content = f"""
        섭외등록 내용이 도착했습니다.

        작성자: {username}
        제목: {title}
        내용: {content}
        법인명: {corporation_name}
        """
        send_email(
            background_tasks,
            "섭외등록 내용이 도착했습니다",
            "sjung8009@naver.com",
            email_content,
        )

    return RedirectResponse(url="/contact", status_code=303)




# 섭외등록 생성 페이지
@router.get("/contact/create")
async def create_post_page(request: Request):
    username = request.session.get("username")
    corporation_name = request.session.get("corporation_name", None)
    return templates.TemplateResponse("contact/contact_create.html", {
        "request": request, 
        "username": username, 
        "corporation_name": corporation_name
    })

# 섭외등록 검색
@router.get("/contact/search")
async def search_posts(
    request: Request,
    search_type: str = Query(...),
    search_query: str = Query(...),
    db: Session = Depends(get_db),
):
    username = request.session.get("username")
    
    if search_type == "title":
        posts = db.query(Post).filter(Post.title.contains(search_query)).all()
    elif search_type == "content":
        posts = db.query(Post).filter(Post.content.contains(search_query)).all()
    elif search_type == "title_content":
        posts = db.query(Post).filter(
            or_(
                Post.title.contains(search_query),
                Post.content.contains(search_query),
            )
        ).all()
    elif search_type == "region_group":
        posts = db.query(Post).filter(Post.region_group_name.contains(search_query)).all()
    elif search_type == "region_headquarter":
        posts = db.query(Post).filter(Post.region_headquarter_name.contains(search_query)).all()
    elif search_type == "branch_office":
        posts = db.query(Post).filter(Post.branch_office_name.contains(search_query)).all()
    elif search_type == "corporation_name":
        posts = db.query(Post).filter(Post.corporation_name.contains(search_query)).all()    
    else:
        posts = db.query(Post).all()

    return templates.TemplateResponse(
        "contact/contact.html", 
        {
            "request": request,
            "posts": posts,
            "username": username
        }
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
        }
    )

# 섭외등록 수정 페이지
@router.get("/contact/update/{post_id}")
async def update_post_page(request: Request, post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        return RedirectResponse(url="/contact", status_code=303)
    
    username = request.session.get("username")
    return templates.TemplateResponse("contact/contact_update.html", {"request": request, "post": post, "username": username})

# 섭외등록 수정
@router.post("/contact/update/{post_id}")
async def update_post(
    request: Request,
    post_id: int,
    title: str = Form(...),
    content: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
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





# 카카오 지도 API
@router.get("/search", response_class=HTMLResponse)
async def get_search_page(request: Request):
    kakao_map_api_key = os.getenv("KAKAO_MAP_API_KEY")
    return templates.TemplateResponse("contact/map.html", {"request": request, "kakao_map_api_key": kakao_map_api_key})


@router.post("/search", response_class=HTMLResponse)
async def search_location(request: Request):
    return templates.TemplateResponse("contact/map.html", {"request": request})



# 파일 다운로드 엔드포인트
@router.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# 채팅 기능 관련
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
            if data.startswith("/w"):
                _, target_username, *message = data.split(" ")
                message = " ".join(message)
                await manager.send_personal_message(
                    f"{username}님으로부터 귓속말: {message}", target_username
                )
            else:
                await manager.broadcast(f"{username}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(username)
        await manager.broadcast(f"{username} 사용자가 채팅에서 퇴장하였습니다.")


# 기능홈페이지
@router.get("/contact4")
async def read_contact(request: Request):
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact4.html", {"request": request, "username": username}
    )

# 비밀번호 가져오기
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")


class PasswordVerification(BaseModel):
    password: str


@router.post("/verify_password2")
async def verify_password2(data: PasswordVerification):
    correct_password = CHAT_PASSWORD

    logging.info(
        f"Received password verification request. Password: {data.password}")
    # logger = logging.getLogger(data.password)
    # logger.debug(data.password)

    result = data.password == correct_password
    logging.info(f"Verifying password. Result: {result}")
    return {"success": result}
