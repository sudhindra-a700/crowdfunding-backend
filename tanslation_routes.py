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

# Import our services
from complete_backend_translation_service import (
    get_translation_service, 
    OptimizedTranslationService,
    SupportedLanguage,
    TranslationRequest,
    TranslationResponse
)
from complete_term_simplification_service import (
    get_simplification_service,
    EnhancedTermSimplificationService,
    ComplexityLevel,
    TermDefinition,
    SimplificationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["translation", "simplification"])

# Pydantic models for API requests/responses
class TranslateTextRequest(BaseModel):
    """Request model for single text translation"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    source_language: str = Field(..., description="Source language code (en, hi, ta, te)")
    target_language: str = Field(..., description="Target language code (en, hi, ta, te)")
    context: Optional[str] = Field(None, max_length=500, description="Optional context for better translation")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()
    
    @validator('source_language', 'target_language')
    def validate_language(cls, v):
        valid_langs = ['en', 'hi', 'ta', 'te']
        if v not in valid_langs:
            raise ValueError(f'Language must be one of: {valid_langs}')
        return v

class BatchTranslateRequest(BaseModel):
    """Request model for batch translation"""
    texts: List[str] = Field(..., min_items=1, max_items=50, description="List of texts to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 5000:
                raise ValueError(f'Text at index {i} exceeds maximum length of 5000 characters')
        
        return [text.strip() for text in v]

class SimplifyTextRequest(BaseModel):
    """Request model for text simplification"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to simplify")
    target_level: str = Field("simple", description="Target complexity level")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()
    
    @validator('target_level')
    def validate_target_level(cls, v):
        valid_levels = ['very_simple', 'simple', 'moderate', 'complex', 'very_complex']
        if v not in valid_levels:
            raise ValueError(f'Target level must be one of: {valid_levels}')
        return v

class TermSearchRequest(BaseModel):
    """Request model for term search"""
    query: str = Field(..., min_length=1, max_length=100, description="Search query")
    category: Optional[str] = Field(None, description="Optional category filter")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")

# Response models
class TranslationResponseModel(BaseModel):
    """Response model for translation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    processing_time: float
    cached: bool

class BatchTranslationResponseModel(BaseModel):
    """Response model for batch translation"""
    translations: List[TranslationResponseModel]
    total_count: int
    total_processing_time: float
    cache_hit_rate: float

class SimplificationResponseModel(BaseModel):
    """Response model for text simplification"""
    original_text: str
    simplified_text: str
    complexity_before: float
    complexity_after: float
    terms_simplified: List[str]
    processing_time: float

class TermDefinitionModel(BaseModel):
    """Response model for term definition"""
    term: str
    simple_definition: str
    detailed_definition: str
    examples: List[str]
    synonyms: List[str]
    category: str
    complexity_level: str
    source: str

class ComplexityAnalysisModel(BaseModel):
    """Response model for complexity analysis"""
    complexity_score: float
    complexity_level: str
    complexity_description: str
    word_count: int
    sentence_count: int
    complex_terms: List[Dict]
    recommendations: List[str]

class ServiceHealthModel(BaseModel):
    """Response model for service health"""
    translation_service: Dict
    simplification_service: Dict
    timestamp: datetime

# Translation endpoints
@router.post("/translate", response_model=TranslationResponseModel)
async def translate_text(
    request: TranslateTextRequest,
    translation_service: OptimizedTranslationService = Depends(get_translation_service)
):
    """
    Translate text from source language to target language
    
    Supports 4 languages: English (en), Hindi (hi), Tamil (ta), Telugu (te)
    """
    try:
        logger.info(f"🔄 Translation request: {request.source_language} -> {request.target_language}")
        
        result = await translation_service.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=request.context
        )
        
        return TranslationResponseModel(
            original_text=result.original_text,
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            cached=result.cached
        )
        
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@router.post("/translate/batch", response_model=BatchTranslationResponseModel)
async def translate_batch(
    request: BatchTranslateRequest,
    background_tasks: BackgroundTasks,
    translation_service: OptimizedTranslationService = Depends(get_translation_service)
):
    """
    Translate multiple texts in batch for better performance
    """
    try:
        logger.info(f"🔄 Batch translation: {len(request.texts)} texts, {request.source_language} -> {request.target_language}")
        
        start_time = asyncio.get_event_loop().time()
        
        results = await translation_service.translate_batch(
            texts=request.texts,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        total_processing_time = asyncio.get_event_loop().time() - start_time
        
        # Calculate cache hit rate
        cached_count = sum(1 for result in results if result.cached)
        cache_hit_rate = cached_count / len(results) if results else 0
        
        translations = [
            TranslationResponseModel(
                original_text=result.original_text,
                translated_text=result.translated_text,
                source_language=result.source_language,
                target_language=result.target_language,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                cached=result.cached
            )
            for result in results
        ]
        
        return BatchTranslationResponseModel(
            translations=translations,
            total_count=len(translations),
            total_processing_time=total_processing_time,
            cache_hit_rate=cache_hit_rate
        )
        
    except Exception as e:
        logger.error(f"❌ Batch translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

@router.get("/translate/languages")
async def get_supported_languages(
    translation_service: OptimizedTranslationService = Depends(get_translation_service)
):
    """
    Get list of supported languages for translation
    """
    try:
        languages = translation_service.get_supported_languages()
        return {
            "supported_languages": languages,
            "total_count": len(languages),
            "language_pairs": len(languages) * (len(languages) - 1)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get languages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported languages")

# Simplification endpoints
@router.post("/simplify", response_model=SimplificationResponseModel)
async def simplify_text(
    request: SimplifyTextRequest,
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Simplify complex text to make it easier to understand
    
    Target levels: very_simple, simple, moderate, complex, very_complex
    """
    try:
        logger.info(f"🔄 Text simplification request: target level {request.target_level}")
        
        # Map string to ComplexityLevel enum
        level_mapping = {
            'very_simple': ComplexityLevel.VERY_SIMPLE,
            'simple': ComplexityLevel.SIMPLE,
            'moderate': ComplexityLevel.MODERATE,
            'complex': ComplexityLevel.COMPLEX,
            'very_complex': ComplexityLevel.VERY_COMPLEX
        }
        
        target_level = level_mapping[request.target_level]
        
        result = await simplification_service.simplify_text(
            text=request.text,
            target_level=target_level
        )
        
        return SimplificationResponseModel(
            original_text=result.original_text,
            simplified_text=result.simplified_text,
            complexity_before=result.complexity_before,
            complexity_after=result.complexity_after,
            terms_simplified=result.terms_simplified,
            processing_time=result.processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Text simplification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text simplification failed: {str(e)}")

@router.post("/simplify/analyze", response_model=ComplexityAnalysisModel)
async def analyze_text_complexity(
    request: SimplifyTextRequest,
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Analyze text complexity and get recommendations for improvement
    """
    try:
        logger.info("🔄 Text complexity analysis request")
        
        analysis = simplification_service.analyze_text_complexity(request.text)
        
        return ComplexityAnalysisModel(**analysis)
        
    except Exception as e:
        logger.error(f"❌ Complexity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Complexity analysis failed: {str(e)}")

@router.get("/simplify/define/{term}", response_model=TermDefinitionModel)
async def get_term_definition(
    term: str,
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Get simple definition for a specific term
    """
    try:
        logger.info(f"🔄 Term definition request: {term}")
        
        term_def = simplification_service.get_term_definition(term)
        
        if not term_def:
            raise HTTPException(status_code=404, detail=f"Term '{term}' not found")
        
        return TermDefinitionModel(
            term=term_def.term,
            simple_definition=term_def.simple_definition,
            detailed_definition=term_def.detailed_definition,
            examples=term_def.examples,
            synonyms=term_def.synonyms,
            category=term_def.category,
            complexity_level=term_def.complexity_level.value[1],
            source=term_def.source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Term definition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get term definition: {str(e)}")

@router.post("/simplify/search", response_model=List[TermDefinitionModel])
async def search_terms(
    request: TermSearchRequest,
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Search for terms by query and optional category
    """
    try:
        logger.info(f"🔄 Term search request: {request.query}")
        
        results = simplification_service.search_terms(
            query=request.query,
            category=request.category
        )
        
        # Limit results
        limited_results = results[:request.limit]
        
        return [
            TermDefinitionModel(
                term=term_def.term,
                simple_definition=term_def.simple_definition,
                detailed_definition=term_def.detailed_definition,
                examples=term_def.examples,
                synonyms=term_def.synonyms,
                category=term_def.category,
                complexity_level=term_def.complexity_level.value[1],
                source=term_def.source
            )
            for term_def in limited_results
        ]
        
    except Exception as e:
        logger.error(f"❌ Term search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Term search failed: {str(e)}")

@router.get("/simplify/categories")
async def get_term_categories(
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Get available term categories and their counts
    """
    try:
        categories = simplification_service.get_term_categories()
        return {
            "categories": categories,
            "total_categories": len(categories),
            "total_terms": sum(categories.values())
        }
    except Exception as e:
        logger.error(f"❌ Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get term categories")

# Health and monitoring endpoints
@router.get("/health", response_model=ServiceHealthModel)
async def get_service_health(
    translation_service: OptimizedTranslationService = Depends(get_translation_service),
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Get health status of translation and simplification services
    """
    try:
        translation_health = translation_service.get_service_health()
        simplification_health = simplification_service.get_service_stats()
        
        return ServiceHealthModel(
            translation_service=translation_health,
            simplification_service=simplification_health,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/stats")
async def get_service_statistics(
    translation_service: OptimizedTranslationService = Depends(get_translation_service),
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Get detailed service statistics and metrics
    """
    try:
        translation_stats = translation_service.get_translation_stats()
        simplification_stats = simplification_service.get_service_stats()
        
        return {
            "translation": translation_stats,
            "simplification": simplification_stats,
            "combined_stats": {
                "total_supported_languages": len(translation_service.get_supported_languages()),
                "total_terms_available": simplification_stats["total_terms"],
                "services_healthy": True,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service statistics")

# Utility endpoints
@router.post("/translate/quick")
async def quick_translate(
    text: str,
    target_language: str,
    source_language: str = "en",
    translation_service: OptimizedTranslationService = Depends(get_translation_service)
):
    """
    Quick translation endpoint for simple use cases
    """
    try:
        result = await translation_service.translate_text(
            text=text,
            source_language=source_language,
            target_language=target_language
        )
        
        return {
            "translated_text": result.translated_text,
            "confidence": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"❌ Quick translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick translation failed: {str(e)}")

@router.post("/simplify/quick")
async def quick_simplify(
    text: str,
    level: str = "simple",
    simplification_service: EnhancedTermSimplificationService = Depends(get_simplification_service)
):
    """
    Quick simplification endpoint for simple use cases
    """
    try:
        level_mapping = {
            'very_simple': ComplexityLevel.VERY_SIMPLE,
            'simple': ComplexityLevel.SIMPLE,
            'moderate': ComplexityLevel.MODERATE
        }
        
        target_level = level_mapping.get(level, ComplexityLevel.SIMPLE)
        result = await simplification_service.simplify_text(text, target_level)
        
        return {
            "simplified_text": result.simplified_text,
            "complexity_improvement": result.complexity_before - result.complexity_after
        }
        
    except Exception as e:
        logger.error(f"❌ Quick simplification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick simplification failed: {str(e)}")

# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Startup event to warm up services
@router.on_event("startup")
async def startup_event():
    """Warm up services on startup"""
    try:
        logger.info("🔄 Warming up translation and simplification services...")
        
        # Initialize services
        translation_service = get_translation_service()
        simplification_service = get_simplification_service()
        
        # Test basic functionality
        await translation_service.translate_text("Hello", "en", "hi")
        await simplification_service.simplify_text("This is a test", ComplexityLevel.SIMPLE)
        
        logger.info("✅ Translation and simplification services ready")
        
    except Exception as e:
        logger.error(f"❌ Service warmup failed: {e}")

# Export router
__all__ = ["router"]

