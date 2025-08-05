# app.py
# This is the refactored main application file.
# It imports and uses the routers from your other modules.

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

# Import the routers from your other files
from oauth_routes import oauth_router
from translation_routes import router as translation_router
from fraud_detection import fraud_router # This router is added in fraud_detection.py

# --- Environment Variable Loading ---
load_dotenv()

# --- Environment Variable Configuration ---
FRONTEND_URL = os.getenv("FRONTEND_BASE_URI", "http://haven-streamlit-frontend.onrender.com")

# --- FastAPI App Initialization ---
app = FastAPI(title="HAVEN Crowdfunding Backend", description="API for the HAVEN crowdfunding platform.")

# --- CORS Middleware ---
origins = [
    FRONTEND_URL,
    "http://localhost:8501",  # Streamlit's default local address
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
# This is where we connect your modular files to the main app
app.include_router(oauth_router)
app.include_router(translation_router)
app.include_router(fraud_router)

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the HAVEN Crowdfunding API!"}

# Health check for deployment services
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
