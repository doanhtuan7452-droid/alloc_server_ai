import re
import socket
import ipaddress
import logging
from urllib.parse import urlparse
from typing import Any, Optional, Dict
from fastapi import HTTPException, status
from app.core.config import settings

logger = logging.getLogger("app.core.security")

# Alphanumeric, hyphens, and underscores (suitable for UUIDs, MD5s, hex IDs, and snake_case IDs)
ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]+$")

# Safe characters for HTTP header values (no newlines, control characters, or Mongo operators)
HEADER_VALUE_PATTERN = re.compile(r"^[a-zA-Z0-9 \t\-\.\:\/\=\_\,\;\+\@\*\?\!\%\(\)]+$")

# IP ranges blocked for SSRF protection
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private class A
    ipaddress.ip_network("172.16.0.0/12"),    # Private class B
    ipaddress.ip_network("192.168.0.0/16"),   # Private class C
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local (Azure Metadata IMDS)
    ipaddress.ip_network("::1/128"),          # IPv6 Loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 Private
    ipaddress.ip_network("fe80::/10")         # IPv6 Link-local
]

def sanitize_mongodb_id(value: Any, name: str = "Identifier") -> str:
    """
    Ensures the ID value is a string and contains only alphanumeric characters or hyphens.
    Raises HTTP 400 Bad Request on failure.
    """
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{name} không được phép trống."
        )
        
    str_value = str(value).strip()
    if not ID_PATTERN.match(str_value):
        logger.warning(f"[Security Guard] Chặn ID không hợp lệ: {name}='{str_value}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{name} chứa các ký tự không hợp lệ hoặc cấu trúc truy vấn không an sau."
        )
        
    return str_value

def validate_dynamic_tools_metadata(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validates dynamic_tools_metadata structure, filtering headers against a whitelist
    and blocking header injection or Mongo operator injection.
    Raises HTTP 400 on violation.
    """
    if not meta:
        return None

    validated_meta = {}
    
    # System keys to check
    system_keys = {"workspaceId", "workspace_id", "userId", "user_id", "contextSignature", "context_signature", "contextsignature"}
    
    for key, value in meta.items():
        # Validate system-level parameters
        if key in system_keys:
            if value is not None:
                # System IDs / signatures should be alphanumeric + hyphens
                str_val = str(value).strip()
                if not ID_PATTERN.match(str_val) and len(str_val) > 0:
                    # Signatures can be slightly longer and contain dots/equals/underscores
                    if not re.match(r"^[a-zA-Z0-9\-\.\_\=\/]+$", str_val):
                        logger.warning(f"[Security Guard] Phát hiện giá trị bối cảnh không an toàn cho {key}: {str_val}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Tham số bối cảnh '{key}' chứa cấu trúc hoặc ký tự không hợp lệ."
                        )
                validated_meta[key] = str_val
            continue

        # If it is not a system key, it must be a dictionary representing tool headers
        if isinstance(value, dict):
            validated_headers = {}
            for h_key, h_val in value.items():
                h_key_lower = h_key.strip().lower()
                
                # Check header name whitelist
                if h_key_lower not in settings.ALLOWED_DYNAMIC_HEADERS:
                    logger.warning(f"[Security Guard] Chặn header không nằm trong whitelist: '{h_key}'")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Header '{h_key}' trong cấu hình tool động không được phép sử dụng."
                    )
                
                # Check header value safety (prevent header injection / control characters)
                str_h_val = str(h_val).strip()
                if not HEADER_VALUE_PATTERN.match(str_h_val):
                    logger.warning(f"[Security Guard] Phát hiện giá trị header không an toàn cho '{h_key}': '{str_h_val}'")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Giá trị của header '{h_key}' chứa ký tự không an toàn."
                    )
                
                validated_headers[h_key] = str_h_val
            validated_meta[key] = validated_headers
        else:
            # Reject any other top-level keys that are not dictionaries or system keys
            logger.warning(f"[Security Guard] Chặn thuộc tính lạ trong metadata: '{key}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cấu trúc metadata '{key}' không hợp lệ."
            )
            
    return validated_meta

def is_trusted_url(url: str) -> bool:
    """
    Checks if the given URL's hostname belongs to the ALLOWED_STORAGE_DOMAINS whitelist.
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if not hostname:
            return False
        
        hostname_lower = hostname.lower()
        for domain in settings.ALLOWED_STORAGE_DOMAINS:
            domain_lower = domain.lower()
            if hostname_lower == domain_lower or hostname_lower.endswith("." + domain_lower):
                return True
        return False
    except Exception:
        return False

def validate_ip_safety(url: str) -> bool:
    """
    Resolves the URL hostname to IP addresses and verifies that none of them belong
    to blocked private, local, or loopback ranges (SSRF mitigation).
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if not hostname:
            return False
            
        # Resolve hostname to IP addresses
        ip_addresses = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in ip_addresses:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            for blocked_range in BLOCKED_IP_RANGES:
                if ip_obj in blocked_range:
                    logger.warning(f"[SSRF Guard] Chặn kết nối tới IP không an toàn: {ip_str} (từ hostname '{hostname}')")
                    return False
        return True
    except Exception as e:
        logger.warning(f"[SSRF Guard] Lỗi phân giải DNS cho hostname '{hostname}': {e}")
        return False
