from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    max_file_size: int = 5 * 1024 * 1024
    tags: list[str] = ["development", "v1.0.0"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
