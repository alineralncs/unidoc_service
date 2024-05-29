from fastapi import FastAPI
import logging 

from routers.users import register_router, login_router, users_router, logout_router
from routers.documents import documents_router
from routers.corpus import corpus_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "UniDoc API is running!"}

app.include_router(register_router)
app.include_router(login_router)
app.include_router(users_router)
app.include_router(logout_router)
app.include_router(documents_router)
app.include_router(corpus_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8049, reload=True)