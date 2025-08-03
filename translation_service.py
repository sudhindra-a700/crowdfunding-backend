"""
HAVEN Crowdfunding Platform - Optimized Translation Service
Supports 4 languages: English, Hindi, Tamil, Telugu
Uses M2M100 model for efficient translation with caching
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from functools import lru_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import redis
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages for translation"""
    ENGLISH = ("en", "English", "🇺🇸")
    HINDI = ("hi", "हिन्दी", "🇮🇳")
    TAMIL = ("ta", "தமிழ்", "🇮🇳")
    TELUGU = ("te", "తెలుగు", "🇮🇳")

@dataclass
class TranslationRequest:
    """Translation request data structure"""
    text: str
    source_language: str
    target_language: str
    context: Optional[str] = None

@dataclass
class TranslationResponse:
    """Translation response data structure"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    processing_time: float
    cached: bool = False

class OptimizedTranslationService:
    """
    Optimized translation service for 4 languages using M2M100-418M model
    Features: Caching, batch processing, quality scoring
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour cache
        self.model_name = "facebook/m2m100_418M"
        self.max_length = 512
        self.batch_size = 8
        
        # Language mapping for M2M100
        self.lang_mapping = {
            "en": "en",
            "hi": "hi", 
            "ta": "ta",
            "te": "te"
        }
        
        self._initialize_cache()
        self._load_model()
    
    def _initialize_cache(self):
        """Initialize Redis cache if available"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache not available: {e}")
            self.redis_client = None
    
    def _load_model(self):
        """Load M2M100 model and tokenizer"""
        try:
            logger.info(f"🔄 Loading translation model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Translation model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load translation model: {e}")
            raise HTTPException(status_code=500, detail="Translation service unavailable")
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        import hashlib
        content = f"{text}:{source_lang}:{target_lang}"
        return f"translation:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_cached_translation(self, cache_key: str) -> Optional[Dict]:
        """Get translation from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _cache_translation(self, cache_key: str, translation_data: Dict):
        """Cache translation result"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(translation_data)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _calculate_confidence_score(self, logits: torch.Tensor) -> float:
        """Calculate confidence score from model logits"""
        try:
            # Get probabilities from logits
            probs = torch.softmax(logits, dim=-1)
            # Calculate average confidence (max probability per token)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            return min(confidence, 1.0)
        except:
            return 0.8  # Default confidence
    
    def _validate_language(self, lang_code: str) -> bool:
        """Validate if language is supported"""
        return lang_code in self.lang_mapping
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for translation"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text.strip()
    
    async def translate_text(
        self, 
        text: str, 
        source_language: str, 
        target_language: str,
        context: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_language: Source language code (en, hi, ta, te)
            target_language: Target language code (en, hi, ta, te)
            context: Optional context for better translation
            
        Returns:
            TranslationResponse with translation details
        """
        start_time = time.time()
        
        # Validate inputs
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not self._validate_language(source_language):
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_language}")
        
        if not self._validate_language(target_language):
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")
        
        # If source and target are same, return original
        if source_language == target_language:
            return TranslationResponse(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence_score=1.0,
                processing_time=time.time() - start_time,
                cached=False
            )
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Check cache
        cache_key = self._get_cache_key(processed_text, source_language, target_language)
        cached_result = self._get_cached_translation(cache_key)
        
        if cached_result:
            logger.info(f"📋 Cache hit for {source_language}->{target_language}")
            return TranslationResponse(
                original_text=text,
                translated_text=cached_result["translated_text"],
                source_language=source_language,
                target_language=target_language,
                confidence_score=cached_result["confidence_score"],
                processing_time=time.time() - start_time,
                cached=True
            )
        
        # Perform translation
        try:
            # Set source language
            self.tokenizer.src_lang = self.lang_mapping[source_language]
            
            # Tokenize input
            inputs = self.tokenizer(
                processed_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id(self.lang_mapping[target_language]),
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode translation
            translated_text = self.tokenizer.batch_decode(
                generated_tokens.sequences, 
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(generated_tokens.scores[0])
            
            # Cache result
            translation_data = {
                "translated_text": translated_text,
                "confidence_score": confidence_score
            }
            self._cache_translation(cache_key, translation_data)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Translation completed: {source_language}->{target_language} in {processing_time:.2f}s")
            
            return TranslationResponse(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence_score=confidence_score,
                processing_time=processing_time,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
    async def translate_batch(
        self, 
        texts: List[str], 
        source_language: str, 
        target_language: str
    ) -> List[TranslationResponse]:
        """
        Translate multiple texts in batch for better performance
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of TranslationResponse objects
        """
        if not texts:
            return []
        
        # Process in batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.translate_text(text, source_language, target_language)
                for text in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch translation error: {result}")
                    # Create error response
                    results.append(TranslationResponse(
                        original_text="",
                        translated_text="Translation failed",
                        source_language=source_language,
                        target_language=target_language,
                        confidence_score=0.0,
                        processing_time=0.0,
                        cached=False
                    ))
                else:
                    results.append(result)
        
        return results
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {
                "code": lang.value[0],
                "name": lang.value[1],
                "flag": lang.value[2],
                "native_name": lang.value[1]
            }
            for lang in SupportedLanguage
        ]
    
    def get_service_health(self) -> Dict[str, any]:
        """Get service health status"""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "cache_available": self.redis_client is not None,
            "supported_languages": len(SupportedLanguage),
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size
        }
    
    def get_translation_stats(self) -> Dict[str, any]:
        """Get translation service statistics"""
        stats = {
            "total_translations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "supported_language_pairs": 0
        }
        
        # Calculate supported language pairs
        num_langs = len(SupportedLanguage)
        stats["supported_language_pairs"] = num_langs * (num_langs - 1)
        
        # Try to get cache stats if Redis is available
        if self.redis_client:
            try:
                # Get cache statistics
                info = self.redis_client.info()
                stats["cache_memory_usage"] = info.get("used_memory_human", "N/A")
                stats["cache_connected_clients"] = info.get("connected_clients", 0)
            except:
                pass
        
        return stats

# Global translation service instance
translation_service = None

def get_translation_service() -> OptimizedTranslationService:
    """Get or create translation service instance"""
    global translation_service
    if translation_service is None:
        translation_service = OptimizedTranslationService()
    return translation_service

# Utility functions for easy integration
@lru_cache(maxsize=100)
def get_language_info(lang_code: str) -> Optional[Dict[str, str]]:
    """Get language information by code"""
    for lang in SupportedLanguage:
        if lang.value[0] == lang_code:
            return {
                "code": lang.value[0],
                "name": lang.value[1],
                "flag": lang.value[2],
                "native_name": lang.value[1]
            }
    return None

def is_supported_language(lang_code: str) -> bool:
    """Check if language is supported"""
    return any(lang.value[0] == lang_code for lang in SupportedLanguage)

async def quick_translate(text: str, target_language: str, source_language: str = "en") -> str:
    """Quick translation function for simple use cases"""
    service = get_translation_service()
    result = await service.translate_text(text, source_language, target_language)
    return result.translated_text

# Example usage and testing
if __name__ == "__main__":
    async def test_translation():
        service = OptimizedTranslationService()
        
        # Test single translation
        result = await service.translate_text(
            "Welcome to HAVEN Crowdfunding Platform", 
            "en", 
            "hi"
        )
        print(f"Translation: {result.translated_text}")
        print(f"Confidence: {result.confidence_score}")
        
        # Test batch translation
        texts = [
            "Create your campaign",
            "Support innovative projects", 
            "Join our community"
        ]
        
        batch_results = await service.translate_batch(texts, "en", "ta")
        for i, result in enumerate(batch_results):
            print(f"Batch {i+1}: {result.translated_text}")
        
        # Test supported languages
        languages = service.get_supported_languages()
        print(f"Supported languages: {languages}")
        
        # Test health check
        health = service.get_service_health()
        print(f"Service health: {health}")
    
    # Run test
    asyncio.run(test_translation())

