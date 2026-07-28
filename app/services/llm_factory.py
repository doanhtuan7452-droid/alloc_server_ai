from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.core.config import settings

class LLMFactory:
    _registry = {
        "openai": ChatOpenAI,
        "gemini": ChatGoogleGenerativeAI,
        "ollama": ChatOllama
    }
    
    # Default models for each provider when model override is not specified
    _default_models = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "ollama": "qwen2.5:7b"
    }

    # Caching dictionary based on (provider, model, temperature)
    _cache = {}

    @classmethod
    def get_llm(cls, provider: str = None, model: str = None, temperature: float = None, **kwargs) -> BaseChatModel:
        prov = (provider or settings.LLM_PROVIDER).lower()
        
        # Fallback to gemini if requested openai but api key is missing
        is_fallback = False
        if prov == "openai" and not settings.OPENAI_API_KEY:
            prov = "gemini"
            model = "gemini-2.0-flash"
            is_fallback = True
            
        # If provider matches system default or fell back, use system model. Otherwise, fall back to provider default.
        if is_fallback or provider is None or prov == settings.LLM_PROVIDER.lower():
            model_name = model or settings.LLM_MODEL
        else:
            model_name = model or cls._default_models.get(prov, "gpt-4o-mini")
            
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        
        # Cache key tuple
        cache_key = (prov, model_name, temp)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        if prov not in cls._registry:
            raise ValueError(f"Unsupported LLM provider: {prov}. Supported: {list(cls._registry.keys())}")

        model_class = cls._registry[prov]

        if prov == "openai":
            instance = model_class(
                model=model_name,
                temperature=temp,
                api_key=settings.OPENAI_API_KEY,
                **kwargs
            )
        elif prov == "gemini":
            instance = model_class(
                model=model_name,
                temperature=temp,
                google_api_key=settings.GEMINI_API_KEY,
                **kwargs
            )
        elif prov == "ollama":
            instance = model_class(
                model=model_name,
                temperature=temp,
                base_url=settings.OLLAMA_BASE_URL,
                **kwargs
            )
            
        cls._cache[cache_key] = instance
        return instance
