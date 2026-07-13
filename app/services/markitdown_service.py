import asyncio
import logging
from markitdown import MarkItDown

logger = logging.getLogger("app.services.markitdown_service")

class MarkItDownService:
    def __init__(self):
        # Khởi tạo đối tượng MarkItDown của Microsoft
        self.md = MarkItDown()

    async def convert_file_to_markdown(self, file_path: str) -> str:
        """
        Chuyển đổi các định dạng file (PDF, DOCX, XLSX, PPTX, HTML, v.v.) sang Markdown.
        Sử dụng asyncio.to_thread để tránh gây block Event Loop của FastAPI.
        """
        logger.info(f"Bắt đầu chuyển đổi tệp {file_path} sang Markdown dùng MarkItDown")
        try:
            # Chạy hàm đồng bộ convert trong một thread pool worker riêng
            result = await asyncio.to_thread(self._sync_convert, file_path)
            logger.info(f"Chuyển đổi Markdown thành công cho {file_path}")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi file {file_path} sang Markdown: {e}")
            raise e

    def _sync_convert(self, file_path: str) -> str:
        """Hàm đồng bộ chạy thực tế MarkItDown convert."""
        res = self.md.convert(file_path)
        return res.text_content

markitdown_service = MarkItDownService()
