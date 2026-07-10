from app.config import settings

print(settings.llm_provider)
print(settings.llm_model)
print(settings.chunk_size)
print(settings.cerebras_api_key[:10])