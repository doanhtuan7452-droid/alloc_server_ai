import uuid
import logging
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.core.exceptions")

async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler that avoids leaking raw tracebacks.
    Preserves HTTPException status codes (such as HTTP 503 for database outages).
    """
    # Preserve explicit HTTPExceptions (like 503, 401, 403, 404)
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        logger.warning(f"HTTPException parsed by global handler: status={exc.status_code}, detail={exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error_reference": str(uuid.uuid4())
            }
        )

    # For other unhandled runtime errors
    error_ref = str(uuid.uuid4())
    logger.error(
        f"[ErrorRef: {error_ref}] Lỗi hệ thống nghiêm trọng xảy ra khi xử lý request {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Đã xảy ra lỗi hệ thống nội bộ. Vui lòng liên hệ bộ phận hỗ trợ kỹ thuật.",
            "error_reference": error_ref
        }
    )

async def custom_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Minimizes Pydantic validation details leakage on HTTP 422 errors.
    """
    error_ref = str(uuid.uuid4())
    logger.warning(f"[Validation ErrorRef: {error_ref}] Lỗi định dạng dữ liệu đầu vào: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Dữ liệu đầu vào không hợp lệ hoặc thiếu các trường bắt buộc.",
            "error_reference": error_ref,
            "errors": [{"field": str(e["loc"][-1]), "message": e["msg"]} for e in exc.errors()]
        }
    )
