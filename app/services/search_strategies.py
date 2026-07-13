from abc import ABC, abstractmethod
from typing import List, Dict, Any
import httpx
import re
import html
import logging
from app.core.config import settings

logger = logging.getLogger("app.services.search_strategies")

class BaseSearchStrategy(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm trên internet và trả về danh sách kết quả dạng:
        [{"title": "...", "link": "...", "snippet": "..."}]
        """
        pass

class GoogleSearchStrategy(BaseSearchStrategy):
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        api_key = settings.GOOGLE_SEARCH_API_KEY
        cse_id = settings.GOOGLE_CSE_ID
        if not api_key or not cse_id:
            logger.warning("Google Search API Key hoặc CSE ID trống. Chuyển hướng sang DuckDuckGo.")
            return await DuckDuckGoSearchStrategy().search(query, max_results)

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": max_results
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    results = []
                    for item in items[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                    return results
                else:
                    logger.error(f"Google Search API trả về mã lỗi {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Lỗi khi gọi Google Search API: {e}")
        
        # Fallback to DuckDuckGo if Google fails
        return await DuckDuckGoSearchStrategy().search(query, max_results)

class TavilySearchStrategy(BaseSearchStrategy):
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        api_key = settings.TAVILY_API_KEY
        if not api_key:
            logger.warning("Tavily API Key trống. Chuyển hướng sang DuckDuckGo.")
            return await DuckDuckGoSearchStrategy().search(query, max_results)

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("results", [])
                    results = []
                    for item in items[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("url", ""),
                            "snippet": item.get("content", "")
                        })
                    return results
                else:
                    logger.error(f"Tavily API trả về mã lỗi {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Lỗi khi gọi Tavily API: {e}")

        # Fallback to DuckDuckGo if Tavily fails
        return await DuckDuckGoSearchStrategy().search(query, max_results)

class DuckDuckGoSearchStrategy(BaseSearchStrategy):
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=10.0)
                if response.status_code != 200:
                    logger.error(f"DuckDuckGo HTML trả về mã lỗi {response.status_code}")
                    return []

                # Sử dụng Regex để parse HTML của DuckDuckGo
                # Mỗi kết quả được bọc bởi các thẻ có class="result__a" cho title/url và class="result__snippet" cho snippet
                html_content = response.text
                
                # Tìm tất cả các thẻ a class="result__a"
                a_pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                a_matches = re.findall(a_pattern, html_content, re.DOTALL)
                
                # Tìm tất cả các thẻ a class="result__snippet"
                snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>'
                snippet_matches = re.findall(snippet_pattern, html_content, re.DOTALL)
                
                results = []
                for idx, (href, title_html) in enumerate(a_matches[:max_results]):
                    # Giải mã HTML Entities và làm sạch text
                    title = html.unescape(re.sub(r'<[^>]+>', '', title_html)).strip()
                    
                    snippet = ""
                    if idx < len(snippet_matches):
                        snippet = html.unescape(re.sub(r'<[^>]+>', '', snippet_matches[idx])).strip()
                    
                    # Trích xuất URL thực từ redirect URL của DDG nếu cần thiết
                    # Link thường có dạng: //duckduckgo.com/l/?kh=-1&uddg=https%3A%2F%2Fpython.org%2F
                    link = href
                    if "uddg=" in href:
                        match = re.search(r'uddg=([^&]+)', href)
                        if match:
                            link = urllib_parse_unquote(match.group(1))
                    elif href.startswith("//"):
                        link = "https:" + href

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                return results
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm trên DuckDuckGo: {e}")
            return []

def urllib_parse_unquote(url_str: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(url_str)

class SearchStrategyFactory:
    @staticmethod
    def get_strategy(provider: str = None) -> BaseSearchStrategy:
        prov = (provider or settings.SEARCH_PROVIDER).lower()
        if prov == "google":
            return GoogleSearchStrategy()
        elif prov == "tavily":
            return TavilySearchStrategy()
        else:
            return DuckDuckGoSearchStrategy()
