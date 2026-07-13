import re
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("app.services.tool_safety_guard")

class ToolSafetyGuard:
    def __init__(self):
        # Các mẫu Regex dùng để phát hiện hành vi Prompt Injection hoặc Jailbreak (cả tiếng Anh và tiếng Việt)
        self.jailbreak_patterns = [
            r"ignore\s+(?:previous\s+)?instructions",
            r"bỏ\s+qua\s+chỉ\s+dẫn",
            r"override\s+safety",
            r"bypass\s+security",
            r"system\s+prompt",
            r"you\s+are\s+now",
            r"lệnh\s+hệ\s+thống",
            r"hãy\s+làm\s+theo",
            r"ignore\s+the\s+safety",
            r"bỏ\s+qua\s+luật",
            r"bỏ\s+qua\s+các\s+bước\s+kiểm\s+tra",
            r"ignore\s+all\s+rules",
            r"bỏ\s+qua\s+mọi\s+quy\s+tắc",
            r"trả\s+về\s+thông\s+tin\s+hệ\s+thống"
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.jailbreak_patterns]

    def check_prompt_injection(self, text: str) -> bool:
        """
        Quét chuỗi văn bản để phát hiện dấu vết tấn công Prompt Injection hoặc Jailbreak.
        Trả về True nếu phát hiện hành vi không an toàn, ngược lại là False.
        """
        if not text:
            return False
        
        # Chuẩn hóa chuỗi văn bản để quét hiệu quả hơn
        text_normalized = text.strip()
        
        for pattern in self.compiled_patterns:
            if pattern.search(text_normalized):
                return True
        return False

    def validate_tool_call(
        self, 
        tool_name: str, 
        args: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Chặn và xác thực toàn bộ các tham số gọi công cụ từ LLM.
        Chỉ tập trung vào quét an toàn (Prompt Injection/Jailbreak) trên các trường kiểu chuỗi.
        Trả về: (is_safe, error_message).
        """
        # 1. Quét tất cả các tham số có giá trị là chuỗi (string)
        for key, value in args.items():
            if isinstance(value, str):
                if self.check_prompt_injection(value):
                    err_msg = f"Phát hiện hành vi Prompt Injection hoặc Jailbreak trong tham số '{key}'."
                    logger.warning(f"[SafetyGuard] Chặn công cụ '{tool_name}' do: {err_msg}")
                    return False, err_msg
            elif isinstance(value, list):
                # Quét đệ quy nếu tham số là danh sách chứa chuỗi
                for idx, item in enumerate(value):
                    if isinstance(item, str) and self.check_prompt_injection(item):
                        err_msg = f"Phát hiện hành vi Prompt Injection hoặc Jailbreak trong phần tử thứ {idx} của tham số '{key}'."
                        logger.warning(f"[SafetyGuard] Chặn công cụ '{tool_name}' do: {err_msg}")
                        return False, err_msg
            elif isinstance(value, dict):
                # Quét các giá trị trong dictionary
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, str) and self.check_prompt_injection(sub_val):
                        err_msg = f"Phát hiện hành vi Prompt Injection hoặc Jailbreak trong thuộc tính '{sub_key}' của tham số '{key}'."
                        logger.warning(f"[SafetyGuard] Chặn công cụ '{tool_name}' do: {err_msg}")
                        return False, err_msg

        return True, None

tool_safety_guard = ToolSafetyGuard()
