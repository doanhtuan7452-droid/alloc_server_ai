import os
import uuid
import logging
import tempfile
from typing import Optional, Dict
import httpx

from app.core.config import settings

logger = logging.getLogger("app.services.file_service")

class FileDownloadService:
    def is_allowed_file(self, file_name: str, file_type: str) -> bool:
        """Kiểm tra định dạng file có được hỗ trợ hay không."""
        # 1. Kiểm tra theo phần mở rộng của file
        _, ext = os.path.splitext(file_name)
        ext = ext.lower().strip()
        if ext in settings.RAG_ALLOWED_EXTENSIONS:
            return True
            
        # 2. Kiểm tra theo MIME type (nếu có)
        mime = file_type.lower().strip()
        for allowed_ext in settings.RAG_ALLOWED_EXTENSIONS:
            # So khớp cơ bản ví dụ: .pdf -> application/pdf
            # Thường chỉ cần kiểm tra đuôi file là đủ và chuẩn nhất
            clean_ext = allowed_ext.replace(".", "")
            if clean_ext in mime:
                return True
                
        return False

    async def download_file(
        self, 
        file_id: str, 
        storage_url: str, 
        dynamic_headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Tải file bất đồng bộ từ Azure Storage hoặc proxy của C# Server.
        Trả về đường dẫn file cục bộ tạm thời.
        """
        # Validate định dạng file từ tên file hoặc URL
        if not self.is_allowed_file(storage_url.split("?")[0].split("/")[-1] or file_id, ""):
            raise ValueError(f"Định dạng tệp không được hỗ trợ để xử lý RAG.")

        temp_dir = tempfile.gettempdir()
        _, ext = os.path.splitext(storage_url.split("?")[0])
        if not ext:
            ext = ".bin" # Fallback extension
        temp_file_name = f"rag_{uuid.uuid4().hex}{ext}"
        temp_file_path = os.path.join(temp_dir, temp_file_name)

        # 1. SSRF Guard: Domain Whitelist Check
        from app.core.security import is_trusted_url, validate_ip_safety
        if not is_trusted_url(storage_url):
            logger.warning(f"[SSRF Guard] Chặn tải từ domain không tin cậy: {storage_url}")
            raise ValueError("Yêu cầu bị chặn: Tên miền không nằm trong danh sách tin cậy.")
        
        # 2. SSRF Guard: IP/DNS Resolution Check
        if not validate_ip_safety(storage_url):
            logger.warning(f"[SSRF Guard] Chặn tải từ IP không an toàn: {storage_url}")
            raise ValueError("Yêu cầu bị chặn: Địa chỉ IP không an toàn.")

        # Parsing dynamic verify setting (CA bundle path or boolean toggle)
        verify_val = settings.CSHARP_STORAGE_CA_BUNDLE
        if verify_val.lower() == "true":
            verify_param = True
        elif verify_val.lower() == "false":
            verify_param = False
        else:
            verify_param = verify_val # Treats as local CA bundle file path

        async with httpx.AsyncClient(verify=verify_param) as client:
            # 3. Disk DoS Guard: Pre-flight HEAD request content-length check
            MAX_ALLOWED_FILE_SIZE = 15 * 1024 * 1024  # 15MB
            try:
                head_headers = {}
                if dynamic_headers:
                    for k, v in dynamic_headers.items():
                        head_headers[k.lower()] = v
                head_resp = await client.request("HEAD", storage_url, headers=head_headers, timeout=5.0)
                if head_resp.status_code == 200:
                    content_length = head_resp.headers.get("Content-Length")
                    if content_length and int(content_length) > MAX_ALLOWED_FILE_SIZE:
                        logger.warning(f"[Disk DoS Guard] Từ chối tải file có dung lượng {content_length} bytes (> 15MB limit)")
                        raise ValueError(f"Kích thước tệp vượt quá giới hạn cho phép ({MAX_ALLOWED_FILE_SIZE // (1024*1024)}MB).")
            except ValueError as val_err:
                raise val_err
            except Exception as net_err:
                logger.warning(f"Không thể kiểm tra kích thước file qua HEAD request: {net_err}")
            # 1. Thử tải trực tiếp từ storage_url (ví dụ: Azure Blob với SAS Token)
            try:
                logger.info(f"Thử tải trực tiếp file {file_id} từ {storage_url}")
                response = await client.get(storage_url, timeout=30.0)
                if response.status_code == 200:
                    with open(temp_file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Tải trực tiếp thành công. Đã lưu tại {temp_file_path}")
                    return temp_file_path
                else:
                    logger.warning(
                        f"Tải trực tiếp thất bại với mã lỗi {response.status_code}. Chuyển sang chế độ Fallback proxy."
                    )
            except Exception as e:
                logger.warning(f"Lỗi khi tải trực tiếp từ {storage_url}: {e}. Chuyển sang chế độ Fallback proxy.")

            # 2. Fallback: Tải thông qua API của C# Server proxy
            csharp_download_url = f"{settings.CSHARP_API_BASE_URL}{settings.CSHARP_FILE_DOWNLOAD_PATH.format(file_id=file_id)}"
            logger.info(f"Đang tải thông qua proxy của C# Server: {csharp_download_url}")
            
            headers = {}
            if dynamic_headers:
                # Trích xuất Authorization từ dynamic_headers gửi từ C#
                for k, v in dynamic_headers.items():
                    headers[k.lower()] = v

            try:
                response = await client.get(csharp_download_url, headers=headers, timeout=30.0)
                if response.status_code == 200:
                    with open(temp_file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Tải file thành công qua proxy C#. Đã lưu tại {temp_file_path}")
                    return temp_file_path
                else:
                    raise Exception(
                        f"Tải file qua proxy thất bại với mã lỗi {response.status_code}: {response.text}"
                    )
            except Exception as proxy_err:
                # Xóa file tạm nếu đã lỡ tạo và gặp lỗi
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
                logger.error(f"Tải file hoàn toàn thất bại: {proxy_err}")
                raise proxy_err

file_service = FileDownloadService()
