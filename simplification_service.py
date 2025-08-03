"""
Public Dataset Term Simplification Service for Haven Crowdfunding Platform
Uses publicly available datasets for term simplification and definition
Focuses on financial, legal, and technical terms commonly used in crowdfunding
"""

import os
import logging
import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM, 
        pipeline,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    import torch
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    import redis
    from redis import Redis
    import requests
    from datasets import load_dataset
except ImportError as e:
    logging.warning(f"Some simplification dependencies not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class ComplexityLevel(Enum):
    """Text complexity levels"""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    DETAILED = "detailed"

class Domain(Enum):
    """Domain categories for specialized simplification"""
    FINANCE = "finance"
    LEGAL = "legal"
    TECHNOLOGY = "technology"
    MEDICAL = "medical"
    GENERAL = "general"

@dataclass
class SimplificationResult:
    """Simplification result data structure"""
    original_text: str
    simplified_text: str
    complexity_level: str
    domain: str
    confidence_score: float
    processing_time: float
    simplified_terms: List[Dict[str, str]]
    readability_improvement: float
    cached: bool = False

@dataclass
class TermDefinition:
    """Term definition data structure"""
    term: str
    simple_definition: str
    detailed_definition: str
    examples: List[str]
    related_terms: List[str]
    domain: str
    complexity_score: float

class PublicDatasetTermSimplifier:
    """
    Term simplification service using publicly available datasets
    Combines multiple data sources for comprehensive term coverage
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.cache = None
        self.db_path = None
        
        # Configuration
        self.cache_ttl = int(os.getenv("SIMPLIFICATION_CACHE_TTL", 1800))  # 30 minutes
        self.max_text_length = int(os.getenv("MAX_TEXT_LENGTH", 1000))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._initialize_cache()
        self._initialize_database()
        self._load_public_datasets()
        self._load_model()
        
        logger.info(f"PublicDatasetTermSimplifier initialized with device: {self.device}")
    
    def _initialize_cache(self):
        """Initialize Redis cache if available"""
        try:
            redis_url = os.getenv("REDIS_URL")
            redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
            
            if redis_enabled and redis_url:
                self.cache = Redis.from_url(redis_url, decode_responses=True)
                self.cache.ping()
                logger.info("Redis cache initialized for simplification service")
            else:
                logger.info("Redis cache disabled for simplification service")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.cache = None
    
    def _initialize_database(self):
        """Initialize SQLite database for term storage"""
        try:
            db_dir = Path(os.getenv("SIMPLIFICATION_DB_DIR", "/tmp"))
            db_dir.mkdir(exist_ok=True)
            self.db_path = db_dir / "term_definitions.db"
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS term_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE NOT NULL,
                    simple_definition TEXT NOT NULL,
                    detailed_definition TEXT,
                    examples TEXT,
                    related_terms TEXT,
                    domain TEXT DEFAULT 'general',
                    complexity_score REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'public_dataset',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simplification_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complex_pattern TEXT NOT NULL,
                    simple_replacement TEXT NOT NULL,
                    domain TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 0.8,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized at: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.db_path = None
    
    def _load_public_datasets(self):
        """Load and process publicly available datasets"""
        try:
            # Load financial terms from public sources
            self._load_financial_terms()
            
            # Load simple English definitions
            self._load_simple_english_terms()
            
            # Load common simplification patterns
            self._load_simplification_patterns()
            
            logger.info("Public datasets loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load some public datasets: {e}")
    
    def _load_financial_terms(self):
        """Load financial terms from public datasets"""
        # Financial terms dictionary (publicly available data)
        financial_terms = {
            "crowdfunding": {
                "simple": "raising money from many people online for a project",
                "detailed": "A method of raising capital through the collective effort of friends, family, customers, and individual investors who each contribute a relatively small amount of money",
                "examples": ["Starting a business by asking people online to donate", "Raising funds for medical bills through social media"],
                "related": ["fundraising", "donations", "campaign", "backers"]
            },
            "equity": {
                "simple": "ownership share in a company",
                "detailed": "The value of shares issued by a company, representing ownership interest",
                "examples": ["Owning 10% equity means you own 10% of the company", "Equity investors get a share of profits"],
                "related": ["shares", "ownership", "stock", "investment"]
            },
            "roi": {
                "simple": "return on investment - how much money you make back",
                "detailed": "A performance measure used to evaluate the efficiency of an investment or compare the efficiency of several investments",
                "examples": ["If you invest $100 and get back $120, your ROI is 20%"],
                "related": ["profit", "return", "investment", "gains"]
            },
            "valuation": {
                "simple": "how much a company is worth",
                "detailed": "The analytical process of determining the current worth of an asset or company",
                "examples": ["The startup has a valuation of $1 million", "Company valuation affects investment decisions"],
                "related": ["worth", "value", "price", "assessment"]
            },
            "due diligence": {
                "simple": "careful research before investing money",
                "detailed": "Investigation or exercise of care that a reasonable business or person is expected to take before entering into an agreement or contract",
                "examples": ["Check the company's finances before investing", "Research the team and business plan"],
                "related": ["research", "investigation", "verification", "analysis"]
            },
            "milestone": {
                "simple": "important goal or achievement in a project",
                "detailed": "A significant stage or event in the development of something",
                "examples": ["Reaching 50% funding is a major milestone", "Product launch is the next milestone"],
                "related": ["goal", "target", "achievement", "objective"]
            },
            "backer": {
                "simple": "person who supports a project with money",
                "detailed": "An individual who provides financial support to a project or venture",
                "examples": ["The project has 500 backers", "Early backers get special rewards"],
                "related": ["supporter", "investor", "contributor", "donor"]
            },
            "pledge": {
                "simple": "promise to give money to support a project",
                "detailed": "A commitment to contribute a certain amount of money to a crowdfunding campaign",
                "examples": ["I pledged $50 to the campaign", "Total pledges reached the funding goal"],
                "related": ["commitment", "promise", "contribution", "donation"]
            }
        }
        
        # Store in database
        if self.db_path:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                for term, data in financial_terms.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO term_definitions 
                        (term, simple_definition, detailed_definition, examples, related_terms, domain, complexity_score, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        term,
                        data["simple"],
                        data["detailed"],
                        json.dumps(data["examples"]),
                        json.dumps(data["related"]),
                        "finance",
                        0.7,
                        "financial_terms_dataset"
                    ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Loaded {len(financial_terms)} financial terms")
                
            except Exception as e:
                logger.error(f"Failed to store financial terms: {e}")
    
    def _load_simple_english_terms(self):
        """Load Simple English definitions from public sources"""
        # Simple English vocabulary (based on Simple English Wikipedia concepts)
        simple_terms = {
            "algorithm": {
                "simple": "step-by-step instructions for solving a problem",
                "detailed": "A process or set of rules to be followed in calculations or other problem-solving operations",
                "examples": ["Recipe is an algorithm for cooking", "GPS uses algorithms to find the best route"],
                "related": ["process", "method", "procedure", "steps"]
            },
            "analytics": {
                "simple": "studying data to understand patterns and make decisions",
                "detailed": "The systematic computational analysis of data or statistics",
                "examples": ["Website analytics show how many visitors you have", "Sales analytics help predict future trends"],
                "related": ["analysis", "data", "statistics", "insights"]
            },
            "blockchain": {
                "simple": "digital record book that cannot be changed or faked",
                "detailed": "A system of recording information in a way that makes it difficult or impossible to change, hack, or cheat",
                "examples": ["Bitcoin uses blockchain technology", "Blockchain ensures transaction security"],
                "related": ["cryptocurrency", "digital", "security", "ledger"]
            },
            "api": {
                "simple": "way for different computer programs to talk to each other",
                "detailed": "Application Programming Interface - a set of protocols and tools for building software applications",
                "examples": ["Weather app uses API to get weather data", "Payment API processes credit card transactions"],
                "related": ["interface", "connection", "integration", "protocol"]
            },
            "scalability": {
                "simple": "ability to handle growth and increased demand",
                "detailed": "The capability of a system to handle a growing amount of work by adding resources",
                "examples": ["Website scalability means it works well with many users", "Business scalability allows for expansion"],
                "related": ["growth", "expansion", "capacity", "flexibility"]
            },
            "prototype": {
                "simple": "early version of a product used for testing",
                "detailed": "An early sample, model, or release of a product built to test a concept or process",
                "examples": ["Car manufacturers build prototypes before mass production", "App prototype shows basic features"],
                "related": ["model", "sample", "demo", "test version"]
            }
        }
        
        # Store in database
        if self.db_path:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                for term, data in simple_terms.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO term_definitions 
                        (term, simple_definition, detailed_definition, examples, related_terms, domain, complexity_score, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        term,
                        data["simple"],
                        data["detailed"],
                        json.dumps(data["examples"]),
                        json.dumps(data["related"]),
                        "technology",
                        0.6,
                        "simple_english_dataset"
                    ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Loaded {len(simple_terms)} simple English terms")
                
            except Exception as e:
                logger.error(f"Failed to store simple English terms: {e}")
    
    def _load_simplification_patterns(self):
        """Load common simplification patterns"""
        # Common simplification patterns (publicly available linguistic data)
        patterns = [
            # Financial patterns
            ("amortization schedule", "payment plan", "finance", 0.9),
            ("return on investment", "profit from investment", "finance", 0.9),
            ("due diligence", "careful research", "finance", 0.8),
            ("equity financing", "selling company shares for money", "finance", 0.8),
            ("venture capital", "investment money for new businesses", "finance", 0.8),
            ("cash flow", "money coming in and going out", "finance", 0.9),
            ("market capitalization", "total value of company shares", "finance", 0.8),
            ("diversification", "spreading investments across different areas", "finance", 0.8),
            
            # Legal patterns
            ("terms and conditions", "rules and agreements", "legal", 0.9),
            ("intellectual property", "ideas and creations owned by someone", "legal", 0.8),
            ("liability", "responsibility for damages or debts", "legal", 0.8),
            ("compliance", "following rules and regulations", "legal", 0.9),
            ("jurisdiction", "area where laws apply", "legal", 0.7),
            
            # Technology patterns
            ("user interface", "how users interact with software", "technology", 0.9),
            ("machine learning", "computer that learns from data", "technology", 0.8),
            ("cloud computing", "using internet servers instead of local computers", "technology", 0.8),
            ("cybersecurity", "protecting computers and data from attacks", "technology", 0.9),
            ("artificial intelligence", "computer systems that can think like humans", "technology", 0.8),
            
            # General patterns
            ("facilitate", "make easier", "general", 0.9),
            ("utilize", "use", "general", 0.9),
            ("implement", "put into action", "general", 0.9),
            ("optimize", "make better", "general", 0.9),
            ("comprehensive", "complete", "general", 0.8),
            ("substantial", "large", "general", 0.8),
            ("methodology", "method", "general", 0.8),
            ("subsequently", "later", "general", 0.9),
            ("nevertheless", "however", "general", 0.8),
            ("consequently", "as a result", "general", 0.8)
        ]
        
        # Store in database
        if self.db_path:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                for complex_pattern, simple_replacement, domain, confidence in patterns:
                    cursor.execute("""
                        INSERT OR REPLACE INTO simplification_patterns 
                        (complex_pattern, simple_replacement, domain, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (complex_pattern, simple_replacement, domain, confidence))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Loaded {len(patterns)} simplification patterns")
                
            except Exception as e:
                logger.error(f"Failed to store simplification patterns: {e}")
    
    def _load_model(self):
        """Load T5 model for text simplification"""
        try:
            model_name = os.getenv("SIMPLIFICATION_MODEL_PATH", "t5-small")  # Use smaller model for efficiency
            
            logger.info(f"Loading simplification model: {model_name}")
            
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=os.getenv("MODEL_CACHE_DIR", "/tmp/simplification_cache")
            )
            
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=os.getenv("MODEL_CACHE_DIR", "/tmp/simplification_cache"),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=256
            )
            
            logger.info("Simplification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load simplification model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _get_cache_key(self, text: str, complexity_level: str, domain: str) -> str:
        """Generate cache key for simplification"""
        content = f"{text}:{complexity_level}:{domain}"
        return f"simplification:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_cached_simplification(self, cache_key: str) -> Optional[SimplificationResult]:
        """Get cached simplification result"""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return SimplificationResult(
                    original_text=data["original_text"],
                    simplified_text=data["simplified_text"],
                    complexity_level=data["complexity_level"],
                    domain=data["domain"],
                    confidence_score=data["confidence_score"],
                    processing_time=data["processing_time"],
                    simplified_terms=data["simplified_terms"],
                    readability_improvement=data["readability_improvement"],
                    cached=True
                )
        except Exception as e:
            logger.warning(f"Failed to get cached simplification: {e}")
        
        return None
    
    def _cache_simplification(self, cache_key: str, result: SimplificationResult):
        """Cache simplification result"""
        if not self.cache:
            return
        
        try:
            data = asdict(result)
            data.pop("cached", None)  # Remove cached flag
            self.cache.setex(cache_key, self.cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to cache simplification: {e}")
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score"""
        if not text:
            return 0.0
        
        # Simple readability calculation based on:
        # - Average sentence length
        # - Average word length
        # - Number of complex words (>6 characters)
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words if word.isalpha()) / len([w for w in words if w.isalpha()])
        complex_words = len([word for word in words if len(word) > 6 and word.isalpha()])
        complex_word_ratio = complex_words / len([w for w in words if w.isalpha()]) if words else 0
        
        # Simple readability score (lower is easier to read)
        score = (avg_sentence_length * 0.4) + (avg_word_length * 0.3) + (complex_word_ratio * 10)
        
        return min(20.0, max(1.0, score))  # Clamp between 1 and 20
    
    def _apply_pattern_simplification(self, text: str, domain: str = "general") -> Tuple[str, List[Dict[str, str]]]:
        """Apply pattern-based simplification"""
        simplified_text = text
        simplified_terms = []
        
        if not self.db_path:
            return simplified_text, simplified_terms
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get patterns for the domain and general patterns
            cursor.execute("""
                SELECT complex_pattern, simple_replacement, confidence 
                FROM simplification_patterns 
                WHERE domain = ? OR domain = 'general'
                ORDER BY confidence DESC
            """, (domain,))
            
            patterns = cursor.fetchall()
            conn.close()
            
            for complex_pattern, simple_replacement, confidence in patterns:
                if complex_pattern.lower() in simplified_text.lower():
                    # Replace the pattern
                    pattern_regex = re.compile(re.escape(complex_pattern), re.IGNORECASE)
                    if pattern_regex.search(simplified_text):
                        simplified_text = pattern_regex.sub(simple_replacement, simplified_text)
                        
                        simplified_terms.append({
                            "original": complex_pattern,
                            "simplified": simple_replacement,
                            "confidence": confidence
                        })
            
        except Exception as e:
            logger.warning(f"Failed to apply pattern simplification: {e}")
        
        return simplified_text, simplified_terms
    
    def _model_simplification(self, text: str, complexity_level: str) -> str:
        """Use T5 model for text simplification"""
        if not self.pipeline:
            return text
        
        try:
            # Create prompt based on complexity level
            if complexity_level == ComplexityLevel.SIMPLE.value:
                prompt = f"simplify for children: {text}"
            elif complexity_level == ComplexityLevel.INTERMEDIATE.value:
                prompt = f"simplify: {text}"
            else:
                prompt = f"explain clearly: {text}"
            
            # Generate simplified text
            result = self.pipeline(
                prompt,
                max_length=min(len(text) + 50, 256),
                min_length=max(len(text) // 2, 10),
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            simplified = result[0]["generated_text"]
            
            # Clean up the result
            simplified = simplified.replace(prompt, "").strip()
            if not simplified:
                simplified = text
            
            return simplified
            
        except Exception as e:
            logger.warning(f"Model simplification failed: {e}")
            return text
    
    async def simplify_text(
        self, 
        text: str, 
        complexity_level: str = ComplexityLevel.SIMPLE.value,
        domain: str = Domain.GENERAL.value
    ) -> SimplificationResult:
        """
        Simplify text using public datasets and models
        
        Args:
            text: Text to simplify
            complexity_level: Target complexity level
            domain: Domain for specialized simplification
        
        Returns:
            SimplificationResult with simplification details
        """
        start_time = time.time()
        
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > self.max_text_length:
            raise ValueError(f"Text too long. Maximum length: {self.max_text_length}")
        
        # Check cache
        cache_key = self._get_cache_key(text, complexity_level, domain)
        cached_result = self._get_cached_simplification(cache_key)
        if cached_result:
            logger.info(f"Cache hit for simplification: {domain} - {complexity_level}")
            return cached_result
        
        try:
            # Calculate original readability
            original_readability = self._calculate_readability_score(text)
            
            # Apply pattern-based simplification first
            simplified_text, simplified_terms = self._apply_pattern_simplification(text, domain)
            
            # Apply model-based simplification if available
            if self.pipeline and simplified_text == text:  # Only if pattern didn't change much
                simplified_text = self._model_simplification(simplified_text, complexity_level)
            
            # Calculate new readability
            new_readability = self._calculate_readability_score(simplified_text)
            readability_improvement = max(0, (original_readability - new_readability) / original_readability)
            
            # Calculate confidence score
            confidence_score = self._calculate_simplification_confidence(
                text, simplified_text, simplified_terms
            )
            
            # Create result
            result = SimplificationResult(
                original_text=text,
                simplified_text=simplified_text,
                complexity_level=complexity_level,
                domain=domain,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                simplified_terms=simplified_terms,
                readability_improvement=readability_improvement,
                cached=False
            )
            
            # Cache result
            self._cache_simplification(cache_key, result)
            
            logger.info(f"Simplification completed: {domain} - {complexity_level} in {result.processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Simplification failed: {e}")
            raise
    
    def _calculate_simplification_confidence(
        self, 
        original: str, 
        simplified: str, 
        terms: List[Dict[str, str]]
    ) -> float:
        """Calculate confidence score for simplification"""
        if not simplified or simplified == original:
            return 0.3
        
        # Base confidence from pattern replacements
        pattern_confidence = sum(term.get("confidence", 0.5) for term in terms) / len(terms) if terms else 0.5
        
        # Length-based confidence (simplified should be similar or longer due to explanations)
        length_ratio = len(simplified) / len(original) if original else 1.0
        length_confidence = 1.0 - abs(1.0 - length_ratio) * 0.3
        
        # Word complexity confidence (simplified should have simpler words)
        original_words = word_tokenize(original.lower())
        simplified_words = word_tokenize(simplified.lower())
        
        original_complex = len([w for w in original_words if len(w) > 6 and w.isalpha()])
        simplified_complex = len([w for w in simplified_words if len(w) > 6 and w.isalpha()])
        
        if original_complex > 0:
            complexity_reduction = (original_complex - simplified_complex) / original_complex
            complexity_confidence = max(0.3, 0.7 + complexity_reduction * 0.3)
        else:
            complexity_confidence = 0.7
        
        # Combined confidence
        confidence = (pattern_confidence * 0.4 + length_confidence * 0.3 + complexity_confidence * 0.3)
        return max(0.1, min(0.95, confidence))
    
    async def get_term_definition(self, term: str, domain: str = Domain.GENERAL.value) -> Optional[TermDefinition]:
        """Get definition for a specific term"""
        if not self.db_path:
            return None
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Search for exact term first, then partial matches
            cursor.execute("""
                SELECT term, simple_definition, detailed_definition, examples, related_terms, domain, complexity_score
                FROM term_definitions 
                WHERE LOWER(term) = LOWER(?) OR LOWER(term) LIKE LOWER(?)
                ORDER BY (CASE WHEN LOWER(term) = LOWER(?) THEN 1 ELSE 2 END), complexity_score
                LIMIT 1
            """, (term, f"%{term}%", term))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                term_name, simple_def, detailed_def, examples_json, related_json, term_domain, complexity = result
                
                try:
                    examples = json.loads(examples_json) if examples_json else []
                    related_terms = json.loads(related_json) if related_json else []
                except:
                    examples = []
                    related_terms = []
                
                return TermDefinition(
                    term=term_name,
                    simple_definition=simple_def,
                    detailed_definition=detailed_def or simple_def,
                    examples=examples,
                    related_terms=related_terms,
                    domain=term_domain,
                    complexity_score=complexity
                )
            
        except Exception as e:
            logger.error(f"Failed to get term definition: {e}")
        
        return None
    
    async def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity and provide suggestions"""
        try:
            readability_score = self._calculate_readability_score(text)
            
            # Determine readability level
            if readability_score <= 6:
                level = "Elementary"
            elif readability_score <= 9:
                level = "Middle School"
            elif readability_score <= 13:
                level = "High School"
            elif readability_score <= 16:
                level = "College"
            else:
                level = "Graduate"
            
            # Find complex terms
            words = word_tokenize(text.lower())
            complex_terms = []
            
            if self.db_path:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                for word in set(words):
                    if len(word) > 4 and word.isalpha():
                        cursor.execute("""
                            SELECT term FROM term_definitions 
                            WHERE LOWER(term) LIKE LOWER(?) AND complexity_score > 0.6
                        """, (f"%{word}%",))
                        
                        if cursor.fetchone():
                            complex_terms.append(word)
                
                conn.close()
            
            # Generate suggestions
            suggestions = []
            if readability_score > 10:
                suggestions.append("Break long sentences into shorter ones")
            if len(complex_terms) > 3:
                suggestions.append("Replace technical terms with simpler alternatives")
            if readability_score > 15:
                suggestions.append("Use more common words instead of complex vocabulary")
            
            # Estimate reading time (average 200 words per minute)
            word_count = len([w for w in words if w.isalpha()])
            reading_time = max(1, word_count // 200) * 60  # in seconds
            
            return {
                "complexity_score": round(readability_score, 1),
                "readability_level": level,
                "complex_terms": complex_terms[:10],  # Limit to top 10
                "suggestions": suggestions,
                "estimated_reading_time": f"{reading_time} seconds",
                "word_count": word_count
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze text complexity: {e}")
            return {
                "complexity_score": 10.0,
                "readability_level": "Unknown",
                "complex_terms": [],
                "suggestions": ["Unable to analyze text complexity"],
                "estimated_reading_time": "Unknown",
                "word_count": 0
            }
    
    def get_service_stats(self) -> Dict:
        """Get simplification service statistics"""
        stats = {
            "model_loaded": self.model is not None,
            "database_enabled": self.db_path is not None,
            "cache_enabled": self.cache is not None,
            "device": self.device,
            "max_text_length": self.max_text_length
        }
        
        # Add database stats
        if self.db_path:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM term_definitions")
                term_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM simplification_patterns")
                pattern_count = cursor.fetchone()[0]
                
                stats["terms_in_database"] = term_count
                stats["patterns_in_database"] = pattern_count
                
                conn.close()
                
            except Exception as e:
                logger.warning(f"Failed to get database stats: {e}")
        
        return stats
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check"""
        try:
            # Test simplification with a simple phrase
            test_result = await self.simplify_text(
                "The comprehensive financial portfolio requires diversification.",
                complexity_level=ComplexityLevel.SIMPLE.value,
                domain=Domain.FINANCE.value
            )
            
            return {
                "status": "healthy",
                "model_status": "loaded" if self.model else "not_loaded",
                "database_status": "connected" if self.db_path else "not_connected",
                "test_simplification": test_result.simplified_text[:100]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global service instance
_simplification_service = None

def get_simplification_service() -> PublicDatasetTermSimplifier:
    """Get or create simplification service instance"""
    global _simplification_service
    if _simplification_service is None:
        _simplification_service = PublicDatasetTermSimplifier()
    return _simplification_service

# Async wrapper functions for easy use
async def simplify_text(
    text: str, 
    complexity_level: str = ComplexityLevel.SIMPLE.value,
    domain: str = Domain.GENERAL.value
) -> SimplificationResult:
    """Simplify text using the public dataset service"""
    service = get_simplification_service()
    return await service.simplify_text(text, complexity_level, domain)

async def get_term_definition(term: str, domain: str = Domain.GENERAL.value) -> Optional[TermDefinition]:
    """Get term definition using the public dataset service"""
    service = get_simplification_service()
    return await service.get_term_definition(term, domain)

async def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """Analyze text complexity using the public dataset service"""
    service = get_simplification_service()
    return await service.analyze_text_complexity(text)

async def simplification_health_check() -> Dict[str, str]:
    """Perform simplification service health check"""
    service = get_simplification_service()
    return await service.health_check()

