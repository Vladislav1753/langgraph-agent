from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    max_file_size: int = 5 * 1024 * 1024
    tags: list[str] = ["development", "v1.0.0"]
    tool_timeout: int = 30
    task_timeout: int = 60
    task_store_maxsize: int = 1000
    task_store_ttl: int = 3600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
