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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    ENGLISH = ("en", "English", "🇺🇸")
    HINDI = ("hi", "हिन्दी", "🇮🇳")
    TAMIL = ("ta", "தமிழ்", "🇮🇳")
    TELUGU = ("te", "తెలుగు", "🇮🇳")

@dataclass
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    context: Optional[str] = None

@dataclass
class TranslationResponse:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    processing_time: float
    cached: bool = False

class OptimizedTranslationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.redis_client = None
        self.cache_ttl = 3600
        self.model_name = "facebook/m2m100_418M"
        self.max_length = 512
        self.batch_size = 8
        self.lang_mapping = {"en": "en", "hi": "hi", "ta": "ta", "te": "te"}
        self._initialize_cache()
        self._load_model()
    
    def _initialize_cache(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache not available: {e}")
            self.redis_client = None
    
    def _load_model(self):
        try:
            logger.info(f"🔄 Loading translation model: {self.model_name}")
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Translation model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load translation model: {e}")
            raise HTTPException(status_code=500, detail="Translation service unavailable")
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        import hashlib
        content = f"{text}:{source_lang}:{target_lang}"
        return f"translation:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_cached_translation(self, cache_key: str) -> Optional[Dict]:
        if not self.redis_client: return None
        try:
            cached = self.redis_client.get(cache_key)
            if cached: return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
    
    def _cache_translation(self, cache_key: str, translation_data: Dict):
        if not self.redis_client: return
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(translation_data))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _calculate_confidence_score(self, logits: torch.Tensor) -> float:
        try:
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            return min(confidence, 1.0)
        except:
            return 0.8
    
    def _validate_language(self, lang_code: str) -> bool:
        return lang_code in self.lang_mapping
    
    def _preprocess_text(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) > 1000:
            text = text[:1000] + "..."
        return text.strip()
    
    async def translate_text(self, text: str, source_language: str, target_language: str, context: Optional[str] = None) -> TranslationResponse:
        start_time = time.time()
        if not text or not text.strip(): raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not self._validate_language(source_language): raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_language}")
        if not self._validate_language(target_language): raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")
        if source_language == target_language:
            return TranslationResponse(original_text=text, translated_text=text, source_language=source_language, target_language=target_language, confidence_score=1.0, processing_time=time.time() - start_time, cached=False)
        
        processed_text = self._preprocess_text(text)
        cache_key = self._get_cache_key(processed_text, source_language, target_language)
        cached_result = self._get_cached_translation(cache_key)
        
        if cached_result:
            logger.info(f"📋 Cache hit for {source_language}->{target_language}")
            return TranslationResponse(original_text=text, translated_text=cached_result["translated_text"], source_language=source_language, target_language=target_language, confidence_score=cached_result["confidence_score"], processing_time=time.time() - start_time, cached=True)
        
        try:
            self.tokenizer.src_lang = self.lang_mapping[source_language]
            inputs = self.tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                generated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.get_lang_id(self.lang_mapping[target_language]), max_length=self.max_length, num_beams=4, early_stopping=True, return_dict_in_generate=True, output_scores=True)
            translated_text = self.tokenizer.batch_decode(generated_tokens.sequences, skip_special_tokens=True)[0]
            confidence_score = self._calculate_confidence_score(generated_tokens.scores[0])
            translation_data = {"translated_text": translated_text, "confidence_score": confidence_score}
            self._cache_translation(cache_key, translation_data)
            processing_time = time.time() - start_time
            logger.info(f"✅ Translation completed: {source_language}->{target_language} in {processing_time:.2f}s")
            return TranslationResponse(original_text=text, translated_text=translated_text, source_language=source_language, target_language=target_language, confidence_score=confidence_score, processing_time=processing_time, cached=False)
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
    async def translate_batch(self, texts: List[str], source_language: str, target_language: str) -> List[TranslationResponse]:
        if not texts: return []
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_tasks = [self.translate_text(text, source_language, target_language) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch translation error: {result}")
                    results.append(TranslationResponse(original_text="", translated_text="Translation failed", source_language=source_language, target_language=target_language, confidence_score=0.0, processing_time=0.0, cached=False))
                else:
                    results.append(result)
        return results
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        return [{"code": lang.value[0], "name": lang.value[1], "flag": lang.value[2], "native_name": lang.value[1]} for lang in SupportedLanguage]
    
    def get_service_health(self) -> Dict[str, any]:
        return {"status": "healthy" if self.model is not None else "unhealthy", "model_loaded": self.model is not None, "device": str(self.device), "cache_available": self.redis_client is not None, "supported_languages": len(SupportedLanguage), "model_name": self.model_name, "max_length": self.max_length, "batch_size": self.batch_size}

translation_service = None
def get_translation_service() -> OptimizedTranslationService:
    global translation_service
    if translation_service is None:
        translation_service = OptimizedTranslationService()
    return translation_service
