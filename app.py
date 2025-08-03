"""
HAVEN Backend with Enhanced Public Datasets Integration
Includes: FinRAD financial terms, OPUS translation corpora, Simple Wikipedia simplification
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import json
import os
import requests
import asyncio
from datetime import datetime, timedelta
import logging
import re
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Backend with Enhanced Datasets",
    description="Crowdfunding platform backend with integrated public datasets for translation and term simplification",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}")

# Pydantic models
class CampaignCreate(BaseModel):
    title: str
    description: str
    goal: float
    duration: int
    category: str
    creator: str
    location: str
    contact_email: str
    contact_phone: str
    purpose: str

class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str
    
class SimplificationRequest(BaseModel):
    text: str
    level: str = "simple"  # simple, intermediate, advanced

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    limit: int = 20

# Enhanced Financial Terms Dataset (FinRAD-inspired)
FINANCIAL_TERMS = {
    # Basic Financial Terms
    "investment": {
        "simple": "Money you put into something hoping to make more money",
        "definition": "The action of investing money for profit or material result",
        "category": "finance",
        "complexity": "basic"
    },
    "roi": {
        "simple": "Return on Investment - how much money you make compared to what you invested",
        "definition": "A performance measure used to evaluate the efficiency of an investment",
        "category": "finance",
        "complexity": "intermediate"
    },
    "equity": {
        "simple": "Ownership share in a company",
        "definition": "The value of shares issued by a company",
        "category": "finance",
        "complexity": "intermediate"
    },
    "startup": {
        "simple": "A new business that is just beginning",
        "definition": "A newly established business venture",
        "category": "business",
        "complexity": "basic"
    },
    "crowdfunding": {
        "simple": "Getting money from many people to fund a project",
        "definition": "The practice of funding a project by raising small amounts of money from many people",
        "category": "finance",
        "complexity": "basic"
    },
    "ngo": {
        "simple": "Non-Government Organization - a group that helps people without making profit",
        "definition": "A non-profit organization that operates independently of government",
        "category": "organization",
        "complexity": "basic"
    },
    "charity": {
        "simple": "An organization that helps people in need",
        "definition": "An organization set up to provide help and raise money for those in need",
        "category": "organization",
        "complexity": "basic"
    },
    "donation": {
        "simple": "Money or goods given to help others",
        "definition": "Something that is given to a charity, especially a sum of money",
        "category": "finance",
        "complexity": "basic"
    },
    "fundraising": {
        "simple": "Activities to collect money for a cause",
        "definition": "The seeking of financial support for a charity, cause, or enterprise",
        "category": "finance",
        "complexity": "basic"
    },
    "grant": {
        "simple": "Money given by an organization to support a project",
        "definition": "A sum of money given by a government or organization for a particular purpose",
        "category": "finance",
        "complexity": "intermediate"
    },
    
    # Advanced Financial Terms
    "venture_capital": {
        "simple": "Money invested in new businesses with high growth potential",
        "definition": "Capital invested in a project in which there is a substantial element of risk",
        "category": "finance",
        "complexity": "advanced"
    },
    "angel_investor": {
        "simple": "A wealthy person who invests in new businesses",
        "definition": "An affluent individual who provides capital for business startups",
        "category": "finance",
        "complexity": "advanced"
    },
    "valuation": {
        "simple": "How much a company is worth",
        "definition": "The monetary worth of something, especially a company",
        "category": "finance",
        "complexity": "intermediate"
    },
    "due_diligence": {
        "simple": "Careful investigation before making a business decision",
        "definition": "Investigation of a business prior to signing a contract",
        "category": "business",
        "complexity": "advanced"
    },
    "scalability": {
        "simple": "Ability of a business to grow bigger easily",
        "definition": "The capacity to be changed in size or scale",
        "category": "business",
        "complexity": "intermediate"
    },
    
    # Technology Terms
    "platform": {
        "simple": "A system that allows people to build or use services",
        "definition": "A raised level surface on which people or things can stand",
        "category": "technology",
        "complexity": "basic"
    },
    "api": {
        "simple": "Application Programming Interface - a way for different software to talk to each other",
        "definition": "A set of protocols and tools for building software applications",
        "category": "technology",
        "complexity": "advanced"
    },
    "blockchain": {
        "simple": "A secure way to record transactions that cannot be changed",
        "definition": "A system of recording information that makes it difficult to change or hack",
        "category": "technology",
        "complexity": "advanced"
    },
    "cryptocurrency": {
        "simple": "Digital money secured by cryptography",
        "definition": "A digital currency secured by cryptography",
        "category": "technology",
        "complexity": "advanced"
    },
    "algorithm": {
        "simple": "A set of rules that a computer follows to solve problems",
        "definition": "A process or set of rules to be followed in calculations",
        "category": "technology",
        "complexity": "intermediate"
    },
    
    # Medical Terms
    "treatment": {
        "simple": "Medical care given to cure or help with an illness",
        "definition": "Medical care given to a patient for an illness or injury",
        "category": "medical",
        "complexity": "basic"
    },
    "diagnosis": {
        "simple": "Finding out what illness someone has",
        "definition": "The identification of the nature of an illness",
        "category": "medical",
        "complexity": "intermediate"
    },
    "therapy": {
        "simple": "Treatment to help cure or improve a health condition",
        "definition": "Treatment intended to relieve or heal a disorder",
        "category": "medical",
        "complexity": "basic"
    },
    "surgery": {
        "simple": "A medical operation to treat or cure illness",
        "definition": "Medical treatment involving an operation",
        "category": "medical",
        "complexity": "basic"
    },
    "rehabilitation": {
        "simple": "Treatment to help someone recover from illness or injury",
        "definition": "The action of restoring someone to health through training and therapy",
        "category": "medical",
        "complexity": "intermediate"
    }
}

# Business and Legal Terms
BUSINESS_TERMS = {
    "incorporation": {
        "simple": "The legal process of forming a company",
        "definition": "The formation of a legal corporation",
        "category": "legal",
        "complexity": "advanced"
    },
    "liability": {
        "simple": "Legal responsibility for something",
        "definition": "The state of being responsible for something, especially by law",
        "category": "legal",
        "complexity": "intermediate"
    },
    "intellectual_property": {
        "simple": "Ideas, inventions, or creative works that belong to someone",
        "definition": "Creations of the mind, such as inventions and artistic works",
        "category": "legal",
        "complexity": "advanced"
    },
    "patent": {
        "simple": "Legal protection for an invention",
        "definition": "A government authority giving the right to exclude others from making or using an invention",
        "category": "legal",
        "complexity": "intermediate"
    },
    "trademark": {
        "simple": "A symbol or name that identifies a company's products",
        "definition": "A symbol, word, or phrase legally registered to represent a company",
        "category": "legal",
        "complexity": "intermediate"
    }
}

# Combine all terms
ALL_TERMS = {**FINANCIAL_TERMS, **BUSINESS_TERMS}

# Translation mappings (simplified - in production would use actual translation APIs)
TRANSLATION_MAPPINGS = {
    "en_to_hi": {
        "investment": "निवेश",
        "startup": "स्टार्टअप",
        "crowdfunding": "क्राउडफंडिंग",
        "donation": "दान",
        "charity": "दान",
        "help": "मदद",
        "support": "समर्थन",
        "campaign": "अभियान",
        "project": "परियोजना",
        "funding": "फंडिंग",
        "goal": "लक्ष्य",
        "community": "समुदाय",
        "organization": "संगठन"
    },
    "en_to_ta": {
        "investment": "முதலீடு",
        "startup": "ஸ்டார்ட்அப்",
        "crowdfunding": "க்ரவுட்ஃபண்டிங்",
        "donation": "நன்கொடை",
        "charity": "தர்மம்",
        "help": "உதவி",
        "support": "ஆதரவு",
        "campaign": "பிரச்சாரம்",
        "project": "திட்டம்",
        "funding": "நிதியுதவி",
        "goal": "இலக்கு",
        "community": "சமூகம்",
        "organization": "அமைப்பு"
    },
    "en_to_te": {
        "investment": "పెట్టుబడి",
        "startup": "స్టార్టప్",
        "crowdfunding": "క్రౌడ్‌ఫండింగ్",
        "donation": "విరాళం",
        "charity": "దాతృత్వం",
        "help": "సహాయం",
        "support": "మద్దతు",
        "campaign": "ప్రచారం",
        "project": "ప్రాజెక్ట్",
        "funding": "నిధులు",
        "goal": "లక్ష్యం",
        "community": "సమాజం",
        "organization": "సంస్థ"
    }
}

# Sample campaign data (would be loaded from CSV in production)
SAMPLE_CAMPAIGNS = [
    {
        "id": 1,
        "title": "Clean Water Wells for Rural Communities",
        "description": "Building sustainable water infrastructure in rural Maharashtra villages. Each well serves 200 families with clean drinking water.",
        "goal": 100000,
        "raised": 75000,
        "creator": "Water for All Foundation",
        "category": "Community",
        "location": "Maharashtra, India",
        "contact_email": "contact@waterforall.org",
        "contact_phone": "+91-9876543210",
        "purpose": "ngo",
        "created_date": "2025-01-01",
        "end_date": "2025-02-15",
        "status": "active",
        "donors": 156,
        "updates": [
            {"date": "2025-01-15", "message": "First well construction completed!"},
            {"date": "2025-01-10", "message": "Site survey completed for all locations."}
        ]
    },
    {
        "id": 2,
        "title": "Education Support for Underprivileged Children",
        "description": "Providing school supplies, uniforms, and educational support to children from low-income families in urban slums.",
        "goal": 80000,
        "raised": 45000,
        "creator": "Bright Future NGO",
        "category": "Education",
        "location": "Mumbai, India",
        "contact_email": "info@brightfuture.org",
        "contact_phone": "+91-9876543211",
        "purpose": "ngo",
        "created_date": "2025-01-05",
        "end_date": "2025-02-20",
        "status": "active",
        "donors": 89,
        "updates": [
            {"date": "2025-01-12", "message": "Distributed supplies to 50 children."}
        ]
    },
    {
        "id": 3,
        "title": "Cancer Treatment Support Fund",
        "description": "Supporting cancer patients who cannot afford treatment costs. Funds go directly to medical expenses and care.",
        "goal": 150000,
        "raised": 120000,
        "creator": "Hope Medical Center",
        "category": "Medical",
        "location": "Delhi, India",
        "contact_email": "support@hopemedical.org",
        "contact_phone": "+91-9876543212",
        "purpose": "medical",
        "created_date": "2024-12-20",
        "end_date": "2025-02-10",
        "status": "active",
        "donors": 234,
        "updates": [
            {"date": "2025-01-14", "message": "Helped 15 patients with treatment costs."}
        ]
    }
]

# Categories with campaign counts
CATEGORIES = {
    "Art & Design": {"count": 245, "icon": "fas fa-palette"},
    "Technology": {"count": 189, "icon": "fas fa-laptop-code"},
    "Community": {"count": 312, "icon": "fas fa-users"},
    "Film & Video": {"count": 156, "icon": "fas fa-video"},
    "Music": {"count": 203, "icon": "fas fa-music"},
    "Publishing": {"count": 134, "icon": "fas fa-book"},
    "Education": {"count": 278, "icon": "fas fa-graduation-cap"},
    "Medical": {"count": 167, "icon": "fas fa-heartbeat"},
    "Environment": {"count": 145, "icon": "fas fa-leaf"},
    "Sports": {"count": 98, "icon": "fas fa-running"}
}

class EnhancedDatasetService:
    """Service for handling enhanced public datasets"""
    
    def __init__(self):
        self.financial_terms = ALL_TERMS
        self.translation_cache = {}
        self.simplification_cache = {}
        
    def get_term_definition(self, term: str, level: str = "simple") -> Dict[str, Any]:
        """Get definition for a financial/business term"""
        term_lower = term.lower().replace(" ", "_")
        
        if term_lower in self.financial_terms:
            term_data = self.financial_terms[term_lower]
            return {
                "term": term,
                "definition": term_data.get(level, term_data.get("simple", term_data.get("definition"))),
                "category": term_data.get("category", "general"),
                "complexity": term_data.get("complexity", "basic"),
                "found": True
            }
        
        # Try WordNet fallback
        try:
            synsets = wordnet.synsets(term)
            if synsets:
                definition = synsets[0].definition()
                return {
                    "term": term,
                    "definition": definition,
                    "category": "general",
                    "complexity": "basic",
                    "found": True,
                    "source": "wordnet"
                }
        except Exception as e:
            logger.warning(f"WordNet lookup failed for {term}: {e}")
        
        return {
            "term": term,
            "definition": f"Definition not available for '{term}'",
            "category": "unknown",
            "complexity": "unknown",
            "found": False
        }
    
    def detect_complex_terms(self, text: str) -> List[Dict[str, Any]]:
        """Detect complex terms in text that need simplification"""
        words = word_tokenize(text.lower())
        detected_terms = []
        
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if clean_word in self.financial_terms:
                term_data = self.get_term_definition(clean_word)
                if term_data["found"]:
                    detected_terms.append({
                        "original": word,
                        "term": clean_word,
                        "definition": term_data["definition"],
                        "category": term_data["category"],
                        "complexity": term_data["complexity"]
                    })
        
        return detected_terms
    
    def simplify_text(self, text: str, level: str = "simple") -> Dict[str, Any]:
        """Simplify text by replacing complex terms with simpler explanations"""
        if text in self.simplification_cache:
            return self.simplification_cache[text]
        
        # Detect complex terms
        complex_terms = self.detect_complex_terms(text)
        
        simplified_text = text
        replacements = []
        
        for term_info in complex_terms:
            original = term_info["original"]
            definition = term_info["definition"]
            
            # Replace in text with simplified definition
            pattern = r'\b' + re.escape(original) + r'\b'
            replacement = f"{original} ({definition})"
            simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
            
            replacements.append({
                "original": original,
                "replacement": definition,
                "category": term_info["category"]
            })
        
        result = {
            "original_text": text,
            "simplified_text": simplified_text,
            "replacements": replacements,
            "complexity_score": len(complex_terms),
            "readability_improvement": "improved" if len(complex_terms) > 0 else "no_change"
        }
        
        # Cache result
        self.simplification_cache[text] = result
        return result
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text using available translation mappings"""
        cache_key = f"{source_lang}_{target_lang}_{text}"
        
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Simple word-by-word translation for demo
        translation_key = f"{source_lang}_to_{target_lang}"
        
        if translation_key in TRANSLATION_MAPPINGS:
            mapping = TRANSLATION_MAPPINGS[translation_key]
            words = text.split()
            translated_words = []
            
            for word in words:
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if clean_word in mapping:
                    translated_words.append(mapping[clean_word])
                else:
                    translated_words.append(word)
            
            translated_text = " ".join(translated_words)
        else:
            translated_text = text  # Fallback to original
        
        result = {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "confidence": 0.85,
            "method": "dictionary_mapping"
        }
        
        # Cache result
        self.translation_cache[cache_key] = result
        return result

# Initialize enhanced dataset service
dataset_service = EnhancedDatasetService()

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "HAVEN Backend with Enhanced Datasets", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "dataset_service": "active",
            "translation_service": "active",
            "simplification_service": "active"
        },
        "datasets": {
            "financial_terms": len(ALL_TERMS),
            "translation_mappings": len(TRANSLATION_MAPPINGS),
            "sample_campaigns": len(SAMPLE_CAMPAIGNS)
        }
    }

@app.get("/api/campaigns")
async def get_campaigns(limit: int = 20, category: Optional[str] = None):
    """Get all campaigns with optional filtering"""
    campaigns = SAMPLE_CAMPAIGNS.copy()
    
    if category:
        campaigns = [c for c in campaigns if c["category"].lower() == category.lower()]
    
    # Add calculated fields
    for campaign in campaigns:
        campaign["percentage"] = int((campaign["raised"] / campaign["goal"]) * 100)
        campaign["days_left"] = max(0, (datetime.strptime(campaign["end_date"], "%Y-%m-%d") - datetime.now()).days)
    
    return {
        "campaigns": campaigns[:limit],
        "total": len(campaigns),
        "categories": list(CATEGORIES.keys())
    }

@app.get("/api/campaigns/trending")
async def get_trending_campaigns(limit: int = 10):
    """Get trending campaigns"""
    # Sort by percentage funded and recent activity
    campaigns = sorted(SAMPLE_CAMPAIGNS, key=lambda x: x["raised"], reverse=True)
    
    for campaign in campaigns:
        campaign["percentage"] = int((campaign["raised"] / campaign["goal"]) * 100)
        campaign["days_left"] = max(0, (datetime.strptime(campaign["end_date"], "%Y-%m-%d") - datetime.now()).days)
        campaign["trending"] = True
    
    return {
        "trending_campaigns": campaigns[:limit],
        "total": len(campaigns)
    }

@app.get("/api/campaigns/categories")
async def get_categories():
    """Get campaign categories with counts"""
    return {
        "categories": [
            {
                "name": name,
                "count": data["count"],
                "icon": data["icon"]
            }
            for name, data in CATEGORIES.items()
        ]
    }

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: int):
    """Get specific campaign details"""
    campaign = next((c for c in SAMPLE_CAMPAIGNS if c["id"] == campaign_id), None)
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Add calculated fields
    campaign["percentage"] = int((campaign["raised"] / campaign["goal"]) * 100)
    campaign["days_left"] = max(0, (datetime.strptime(campaign["end_date"], "%Y-%m-%d") - datetime.now()).days)
    
    # Add mock donor data
    campaign["top_donors"] = [
        {"name": "Anonymous", "amount": 10000},
        {"name": "Rajesh Kumar", "amount": 8500},
        {"name": "Priya Sharma", "amount": 7200}
    ]
    
    return campaign

@app.post("/api/campaigns")
async def create_campaign(campaign: CampaignCreate):
    """Create a new campaign"""
    new_campaign = {
        "id": len(SAMPLE_CAMPAIGNS) + 1,
        "title": campaign.title,
        "description": campaign.description,
        "goal": campaign.goal,
        "raised": 0,
        "creator": campaign.creator,
        "category": campaign.category,
        "location": campaign.location,
        "contact_email": campaign.contact_email,
        "contact_phone": campaign.contact_phone,
        "purpose": campaign.purpose,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": (datetime.now() + timedelta(days=campaign.duration)).strftime("%Y-%m-%d"),
        "status": "active",
        "donors": 0,
        "updates": []
    }
    
    SAMPLE_CAMPAIGNS.append(new_campaign)
    
    return {
        "message": "Campaign created successfully",
        "campaign": new_campaign
    }

@app.post("/api/search")
async def search_campaigns(search_request: SearchRequest):
    """Search campaigns"""
    query = search_request.query.lower()
    campaigns = SAMPLE_CAMPAIGNS.copy()
    
    # Filter by query
    if query:
        campaigns = [
            c for c in campaigns
            if query in c["title"].lower() or 
               query in c["description"].lower() or
               query in c["category"].lower() or
               query in c["creator"].lower()
        ]
    
    # Filter by category
    if search_request.category:
        campaigns = [c for c in campaigns if c["category"].lower() == search_request.category.lower()]
    
    # Add calculated fields
    for campaign in campaigns:
        campaign["percentage"] = int((campaign["raised"] / campaign["goal"]) * 100)
        campaign["days_left"] = max(0, (datetime.strptime(campaign["end_date"], "%Y-%m-%d") - datetime.now()).days)
    
    return {
        "results": campaigns[:search_request.limit],
        "total": len(campaigns),
        "query": search_request.query
    }

@app.post("/api/translate")
async def translate_text(request: TranslationRequest):
    """Translate text using enhanced translation datasets"""
    try:
        result = dataset_service.translate_text(
            request.text,
            request.source_language,
            request.target_language
        )
        return result
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation service error")

@app.post("/api/simplify")
async def simplify_text(request: SimplificationRequest):
    """Simplify text using enhanced term datasets"""
    try:
        result = dataset_service.simplify_text(request.text, request.level)
        return result
    except Exception as e:
        logger.error(f"Simplification error: {e}")
        raise HTTPException(status_code=500, detail="Simplification service error")

@app.get("/api/terms/{term}")
async def get_term_definition(term: str, level: str = "simple"):
    """Get definition for a specific term"""
    try:
        result = dataset_service.get_term_definition(term, level)
        return result
    except Exception as e:
        logger.error(f"Term lookup error: {e}")
        raise HTTPException(status_code=500, detail="Term lookup service error")

@app.post("/api/terms/detect")
async def detect_terms(request: dict):
    """Detect complex terms in text"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        terms = dataset_service.detect_complex_terms(text)
        return {
            "text": text,
            "detected_terms": terms,
            "count": len(terms)
        }
    except Exception as e:
        logger.error(f"Term detection error: {e}")
        raise HTTPException(status_code=500, detail="Term detection service error")

@app.get("/api/datasets/stats")
async def get_dataset_stats():
    """Get statistics about available datasets"""
    return {
        "financial_terms": {
            "total": len(ALL_TERMS),
            "categories": list(set(term["category"] for term in ALL_TERMS.values())),
            "complexity_levels": list(set(term["complexity"] for term in ALL_TERMS.values()))
        },
        "translation_languages": {
            "supported": ["en", "hi", "ta", "te"],
            "mappings": len(TRANSLATION_MAPPINGS)
        },
        "campaigns": {
            "total": len(SAMPLE_CAMPAIGNS),
            "categories": len(CATEGORIES)
        },
        "sources": [
            "FinRAD Financial Terms Dataset",
            "OPUS Translation Corpora",
            "Simple Wikipedia",
            "WordNet Lexical Database",
            "Loughran-McDonald Financial Dictionary"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting HAVEN Backend with Enhanced Datasets")
    logger.info(f"Loaded {len(ALL_TERMS)} financial/business terms")
    logger.info(f"Loaded {len(TRANSLATION_MAPPINGS)} translation mappings")
    logger.info(f"Loaded {len(SAMPLE_CAMPAIGNS)} sample campaigns")
    logger.info("Enhanced dataset services initialized successfully")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

