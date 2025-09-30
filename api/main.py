from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import predict as predict_endpoints
from api.endpoints import reference as reference_endpoints
from api.endpoints import chatbot as chatbot_endpoints
from api.endpoints import suitability as suitability_endpoints

app = FastAPI(
    title="FarmEasy API",
    description="API for crop recommendation system with AI-powered suggestions and streaming chatbot",
    version="1.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(
    predict_endpoints.router,
    prefix="/api/v1",
    tags=["predictions"]
)

app.include_router(
    chatbot_endpoints.router,
    prefix="/api/v1",
    tags=["chatbot"]
)

app.include_router(
    reference_endpoints.router,
    prefix="/api/v1",
    tags=["crop-reference"]
)

app.include_router(
    suitability_endpoints.router,
    prefix="/api/v1",
    tags=["crop-suitability"],
    responses={
        400: {"description": "Invalid crop or parameters"},
        503: {"description": "AI service unavailable"}
    }
)

@app.get("/")
def read_root():
    return {
        "message": "FarmEasy API is running",
        "version": app.version,
        "endpoints": {
            "predictions": "/api/v1/predict",
            "crop-reference": "/api/v1/reference",
            "crop-suitability": "/api/v1/check-suitability",
            "chatbot": "/api/v1/chat",
            "chatbot-streaming": "/api/v1/chat/stream",
            "chatbot-streaming-jsonl": "/api/v1/chat/stream-jsonl"
        }
    }