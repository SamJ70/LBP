from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, models, processes

app = FastAPI(
    title="Machining Optimization API",
    description="ML-based parameter & energy optimization for machining processes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(processes.router, prefix="/api/processes", tags=["Processes"])

@app.get("/")
def root():
    return {"message": "Machining Optimization API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}