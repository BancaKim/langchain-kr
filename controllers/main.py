from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from database import Base, engine
from controllers.common_controllers import router
from controllers.baro_controllers import baro
from fastapi.staticfiles import StaticFiles
from controllers.credit_controller import creditreview

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
Base.metadata.create_all(bind=engine)


app.include_router(router)
app.include_router(baro)

templates = Jinja2Templates(directory="templates")

app.mount("/css", StaticFiles(directory="static/css"), name="static")
app.mount("/images", StaticFiles(directory="static/images"), name="static")
app.mount("/js", StaticFiles(directory="static/js"), name="static")
app.mount("/fonts", StaticFiles(directory="static/fonts"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root(request: Request):
    username = request.session.get("username")
    return templates.TemplateResponse(
        "index.html", {"request": request, "username": username}
    )
