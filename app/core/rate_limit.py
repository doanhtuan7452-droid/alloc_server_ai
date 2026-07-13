import os
import logging
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

logger = logging.getLogger("app.core.rate_limit")

def get_rate_limit_key(request: Request) -> str:
    """
    Custom rate limit key:
    1. Looks up 'user_id' or 'conversation_id' in cached request.state.body_json
    2. Fallback to API Key / Auth header
    3. Fallback to remote IP
    """
    # 1. Look in cached JSON body (set by body caching middleware)
    body = getattr(request.state, "body_json", {})
    if isinstance(body, dict):
        user_id = body.get("user_id") or body.get("userId")
        conversation_id = body.get("conversation_id") or body.get("conversationId")
        key = user_id or conversation_id
        if key:
            return f"user:{key}"

    # 2. Look in X-API-Key or Authorization headers
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if api_key:
        return f"apikey:{api_key}"

    # 3. Fallback to IP address
    return f"ip:{get_remote_address(request)}"

# In production, Redis is mandatory
redis_url = settings.REDIS_URL
if settings.ENV == "production":
    if not redis_url:
        logger.critical("CRITICAL: REDIS_URL is not configured in production environment! Rate limit requires a persistent store.")
        raise ValueError("CRITICAL CONFIGURATION ERROR: REDIS_URL environment variable is missing on Production. Startup aborted.")
    else:
        logger.info(f"Initializing SlowAPI Limiter with Redis storage backend: {redis_url}")
        limiter = Limiter(key_func=get_rate_limit_key, storage_uri=redis_url)
else:
    if redis_url:
        logger.info(f"Initializing SlowAPI Limiter with Redis storage backend (Dev): {redis_url}")
        limiter = Limiter(key_func=get_rate_limit_key, storage_uri=redis_url)
    else:
        logger.info("Initializing SlowAPI Limiter with In-Memory storage (Dev fallback)")
        limiter = Limiter(key_func=get_rate_limit_key)
