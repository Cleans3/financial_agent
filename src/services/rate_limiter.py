"""
Rate Limiting Service for API endpoints
Implements per-user and per-IP rate limiting with token bucket algorithm
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple in-memory rate limiter using token bucket algorithm"""
    
    def __init__(self, requests: int, window_seconds: int):
        """
        Initialize rate limiter
        
        Args:
            requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        self.requests = requests
        self.window_seconds = window_seconds
        self.buckets = defaultdict(lambda: {"tokens": requests, "last_update": time.time()})
    
    def is_allowed(self, key: str) -> Tuple[bool, dict]:
        """
        Check if request is allowed for the given key (user_id or IP)
        
        Returns:
            Tuple of (allowed: bool, info: dict with remaining requests and retry_after)
        """
        now = time.time()
        bucket = self.buckets[key]
        
        # Add tokens based on elapsed time
        elapsed = now - bucket["last_update"]
        new_tokens = (elapsed / self.window_seconds) * self.requests
        bucket["tokens"] = min(self.requests, bucket["tokens"] + new_tokens)
        bucket["last_update"] = now
        
        # Check if request is allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, {
                "remaining": int(bucket["tokens"]),
                "retry_after": None
            }
        else:
            # Calculate when the next token will be available
            retry_after = (1 - bucket["tokens"]) * (self.window_seconds / self.requests)
            return False, {
                "remaining": 0,
                "retry_after": int(retry_after) + 1
            }


# Global rate limiters
login_limiter = RateLimiter(requests=5, window_seconds=900)  # 5 attempts per 15 minutes
api_limiter = RateLimiter(requests=100, window_seconds=60)  # 100 requests per minute


def rate_limit_login(user_identifier: str) -> Tuple[bool, dict]:
    """Rate limit for login endpoint (per username or IP)"""
    return login_limiter.is_allowed(f"login:{user_identifier}")


def rate_limit_api(user_id: str) -> Tuple[bool, dict]:
    """Rate limit for general API endpoints (per user_id)"""
    return api_limiter.is_allowed(f"api:{user_id}")


def rate_limit_ip(ip_address: str) -> Tuple[bool, dict]:
    """Rate limit for general API endpoints (per IP address)"""
    return api_limiter.is_allowed(f"ip:{ip_address}")


def cleanup_old_buckets(limiter: RateLimiter, max_age_seconds: int = 3600):
    """Clean up old bucket entries to prevent memory leaks"""
    now = time.time()
    expired_keys = [
        key for key, bucket in limiter.buckets.items()
        if (now - bucket["last_update"]) > max_age_seconds
    ]
    for key in expired_keys:
        del limiter.buckets[key]
    
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit buckets")
