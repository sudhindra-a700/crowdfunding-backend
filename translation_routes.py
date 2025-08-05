"""
HAVEN Crowdfunding Platform - Translation & Simplification API Routes
Complete FastAPI routes for 4-language translation and term simplification
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from starlette.responses import JSONResponse

from translation_service import get_translation_service, OptimizedTranslationService
from simplification_service import get_simplification_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["translation", "simplification"])

class TranslateTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source_language: str
    target_language: str
    context: Optional[str] = None

# Add other Pydantic models as needed...

@router.post("/translate")
async def translate_text_endpoint(
    request: TranslateTextRequest,
    translation_service: OptimizedTranslationService = Depends(get_translation_service)
):
    try:
        result = await translation_service.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Add other endpoints from your file...
