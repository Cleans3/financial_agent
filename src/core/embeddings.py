from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import hashlib
from enum import Enum
from sentence_transformers import SentenceTransformer
import numpy as np
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Track first embedding model use
_embedding_first_use = True


class EmbeddingModelType(str, Enum):
    FIN_E5_SMALL = "fin-e5-small"
    FIN_E5_BASE = "fin-e5-base"
    FINBERT = "finbert"
    DEFAULT = "default"


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass


class SingleDenseEmbedding(EmbeddingStrategy):
    def __init__(self, model_name: str = None, model_type: str = "general"):
        global _embedding_first_use
        self.model_type = model_type
        if model_name:
            self.model_name = model_name
        else:
            if model_type == "financial":
                self.model_name = self._map_model_name(settings.EMBEDDING_MODEL_FINANCIAL)
            else:
                self.model_name = self._map_model_name(settings.EMBEDDING_MODEL_GENERAL)
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model ({model_type}): {self.model_name}, dim={self._dim}")
            
            # Log first actual use of embedding model
            if _embedding_first_use:
                _embedding_first_use = False
                separator = "x" * 50
                logger.info(separator)
                logger.info("EMBEDDING MODEL ACTIVATION - FIRST USE")
                logger.info(f"Model: {self.model_name}")
                logger.info(f"Type: {model_type.upper()}")
                logger.info(f"Dimension: {self._dim}")
                logger.info(f"Strategy: {settings.CHUNK_EMBEDDING_STRATEGY}")
                logger.info(separator)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}, falling back to default")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dim = 384
    
    def _map_model_name(self, embedding_model: str) -> str:
        mapping = {
            "fin-e5-small": "sentence-transformers/Fin-E5-small",
            "fin-e5-base": "sentence-transformers/Fin-E5-base",
            "finbert": "yiyanghkust/finbert-tone",
            "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
        }
        return mapping.get(embedding_model, "all-MiniLM-L6-v2")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
    
    def embed_single(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query string - alias for embed_single for compatibility"""
        return self.embed_single(query)
    
    @property
    def embedding_dimension(self) -> int:
        return self._dim


class MultiDimensionalEmbedding(EmbeddingStrategy):
    def __init__(self, base_model_name: str = None, model_type: str = "general"):
        self.base_embedding = SingleDenseEmbedding(model_name=base_model_name, model_type=model_type)
        self.metric_categories = {
            "profitability": ["net_income", "revenue", "ebit", "net_margin", "roa", "roe", "eps"],
            "liquidity": ["current_ratio", "quick_ratio", "cash_ratio", "working_capital", "debt_to_equity"],
            "efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover", "days_sales"],
            "growth": ["revenue_growth", "earnings_growth", "asset_growth", "equity_growth"],
            "risk": ["debt_ratio", "interest_coverage", "default_risk", "volatility", "beta"]
        }
    
    def _categorize_text(self, text: str) -> Dict[str, str]:
        categorized = {"general": text}
        for category, keywords in self.metric_categories.items():
            metric_text = " ".join([line for line in text.split('\n') if any(kw in line.lower() for kw in keywords)])
            if metric_text:
                categorized[category] = metric_text
        return categorized
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for text in texts:
            all_embeddings.append(self.embed_single(text))
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        categorized = self._categorize_text(text)
        dimension_embeddings = []
        
        for category in sorted(categorized.keys()):
            cat_text = categorized[category]
            emb = self.base_embedding.embed_single(cat_text)
            dimension_embeddings.extend(emb)
        
        return dimension_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        num_dimensions = len(self.metric_categories) + 1
        return self.base_embedding.embedding_dimension * num_dimensions


class HierarchicalEmbedding(EmbeddingStrategy):
    def __init__(self, base_model_name: str = None, chunk_size: int = None, model_type: str = "general"):
        self.base_embedding = SingleDenseEmbedding(model_name=base_model_name, model_type=model_type)
        self.chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = text.replace('।', '.').split('.')
        return [s.strip() for s in sentences if s.strip()]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_single(text) for text in texts]
    
    def embed_single(self, text: str) -> List[float]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return self.base_embedding.embed_single(text)
        
        sentence_embeddings = self.base_embedding.embed(sentences)
        chunk_embeddings = []
        
        for i in range(0, len(sentence_embeddings), max(1, len(sentence_embeddings) // 3)):
            chunk_sents = sentence_embeddings[i:i+max(1, len(sentence_embeddings)//3)]
            chunk_emb = np.mean(chunk_sents, axis=0).tolist()
            chunk_embeddings.extend(chunk_emb)
        
        sentence_avg = np.mean(sentence_embeddings, axis=0).tolist()
        chunk_embeddings.extend(sentence_avg)
        
        return chunk_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        return self.base_embedding.embedding_dimension * 4


_embedding_cache = {}

def get_embedding_strategy(strategy: str = None, model_type: str = "general") -> EmbeddingStrategy:
    strategy = strategy or settings.CHUNK_EMBEDDING_STRATEGY
    cache_key = f"{strategy}_{model_type}"
    
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    if strategy == "multi-dimensional":
        strategy_instance = MultiDimensionalEmbedding(model_type=model_type)
    elif strategy == "hierarchical":
        strategy_instance = HierarchicalEmbedding(model_type=model_type)
    else:
        strategy_instance = SingleDenseEmbedding(model_type=model_type)
    
    _embedding_cache[cache_key] = strategy_instance
    return strategy_instance
