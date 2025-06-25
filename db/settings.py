from os import getenv
from typing import Optional

# ① import SettingsConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.log import logger


class DbSettings(BaseSettings):
    # ② tell Pydantic where to load and what to ignore
    model_config = SettingsConfigDict(
        env_file=".env",           # ← loads .env into this settings class
        env_file_encoding="utf-8",
        env_prefix="DB_",          # ← only read variables that start with DB_
        extra="ignore",            # ← drop everything else (DATABASE_URL, OPENAI_API_KEY, etc)
    )

    # Database configuration, now coming from DB_HOST, DB_PORT, DB_USER, etc
    db_host: Optional[str]
    db_port: Optional[int]
    db_user: Optional[str]
    db_pass: Optional[str]
    db_database: Optional[str]
    db_driver: str = "postgresql+psycopg"
    migrate_db: bool = False

    def get_db_url(self) -> str:
        db_url = (
            f"{self.db_driver}://{self.db_user}"
            f"{f':{self.db_pass}' if self.db_pass else ''}"
            f"@{self.db_host}:{self.db_port}/{self.db_database}"
        )

        if "None" in db_url and getenv("RUNTIME_ENV") is None:
            from workspace.dev_resources import dev_db

            logger.debug("Using local connection")
            local = dev_db.get_db_connection_local()
            if local:
                db_url = local

        if "None" in db_url:
            raise ValueError("Could not build database connection")

        return db_url


# finally instantiate
db_settings = DbSettings()
