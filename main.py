from fastapi import FastAPI
from routes import router

app = FastAPI(
    title="KYC Extraction API",
    version="1.0"
)

app.include_router(router)