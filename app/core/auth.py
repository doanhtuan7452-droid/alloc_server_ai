import secrets
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.core.config import settings

async def get_api_key(x_api_key: Optional[str] = Header(None, description="API Key xác thực giữa C# và Python server")) -> str:
    if not x_api_key or x_api_key != settings.PYTHON_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Không thể xác thực: API Key (X-API-Key) không chính xác hoặc bị thiếu."
        )
    return x_api_key

async def verify_internal_access(
    request: Request,
    x_internal_secret: Optional[str] = Header(None, description="Secret Key xác thực nội bộ giữa C# và Python"),
    x_internal_token: Optional[str] = Header(None, description="Token Key xác thực nội bộ giữa C# và Python (gửi từ client C#)")
) -> str:
    # 1. IP Whitelisting (Defense-in-Depth)
    client_host = request.client.host if request.client else None
    if not client_host or (client_host not in settings.ALLOWED_INTERNAL_IPS and client_host != "testclient"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found"
        )
        
    # Lấy token từ header X-Internal-Token hoặc X-Internal-Secret để đảm bảo tương thích
    secret = x_internal_token or x_internal_secret
        
    # 2. PSK Checking (Timing Attack Resistant)
    if not secret or not secrets.compare_digest(secret, settings.INTERNAL_API_SECRET):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found"
        )
        
    return secret
