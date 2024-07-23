from email.mime.text import MIMEText
import os
import shutil
import smtplib
import time
from fastapi import (
    APIRouter,
    BackgroundTasks,
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
from sqlalchemy import or_
from sqlalchemy.orm import Session
from models.common_models import Contact, Post, User, Notice, Qna, Reply
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

router = APIRouter()
templates = Jinja2Templates(directory="templates")



# 파일 업로드 관련

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


def basename(value):
    return os.path.basename(value)


templates.env.filters["basename"] = basename

#로그인화면이동
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
    existing_user = db.query(User).filter(User.username == signup_data.username).first()
    if existing_user:
        return JSONResponse(
            status_code=400, content={"message": "이미 동일 사용자 이름이 가입되어 있습니다.", "message_icon": "error"}
        )
    hashed_password = get_password_hash(signup_data.password)
    new_user = User(
        username=signup_data.username,
        email=signup_data.email,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    try:
        db.commit()
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": "회원가입이 실패했습니다. 기입한 내용을 확인해보세요.", "message_icon": "error"}
        )
    db.refresh(new_user)
    return JSONResponse(
        status_code=200, content={"message": "회원가입이 성공했습니다.", "message_icon": "success", "url": "/login"}
    )




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
            {"request": request, "message": "로그인이 성공했습니다.", "message_icon": "success", "url": "/"},
        )
        encoded_username = urllib.parse.quote(request.session["username"])  # URL 인코딩
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
        {"request": request, "message": "로그아웃되었습니다.", "message_icon": "success", "url": "/"},
    )
    response.delete_cookie("session")
    return response


# 공지사항 목록 조회
@router.get("/notices")
async def list_notices(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    notices = db.query(Notice).all()
    return templates.TemplateResponse(
        "notice/notice.html", {"request": request, "notices": notices, "username": username}
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
        "notice/notice.html", {"request": request, "notices": notices, "username": username}
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
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
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
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
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


# 섭외등록 페이지
@router.get("/contact2")
async def read_contact(request: Request):
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact2.html", {"request": request, "username": username}
    )


# 섭외등록 이메일 보내기
@router.post("/contact2")
async def submit_contact_form(
    request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...),
    db: Session = Depends(get_db),
):
    send_email(
        background_tasks,
        "Contact Form Submission",
        "sjung8009@naver.com",
        f"Name: {name}\nEmail: {email}\nMessage: {message}",
    )
    return templates.TemplateResponse(
        "contact/contact2.html",
        {"request": request, "message": "Contact form submitted successfully"},
    )


# 파일첨부 게시판형식 (섭외등록)
@router.get("/contact")
async def create_post_page(request: Request, db: Session = Depends(get_db)):
    posts = db.query(Post).all()
    username = request.session.get("username")
    return templates.TemplateResponse(
        "contact/contact.html", {"request": request, "posts": posts, "username": username}
    )


@router.post("/contact")
async def create_post(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    file_path = None
    if file and file.filename:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail="파일 업로드에 실패했습니다.")

    new_post = Post(title=title, content=content, file_path=file_path)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return RedirectResponse(url="/contact", status_code=303)

# 카카오 지도 API
@router.get("/search", response_class=HTMLResponse)
async def get_search_page(request: Request):
    return templates.TemplateResponse("contact/map.html", {"request": request, "keyword": ""})

@router.post("/search", response_class=HTMLResponse)
async def search_location(request: Request, keyword: str = Form(...)):
    return templates.TemplateResponse("contact/map.html", {"request": request, "keyword": keyword})



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
        
        
