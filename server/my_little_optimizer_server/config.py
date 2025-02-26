from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_path: str = "sweeps.db"
    debug: bool = False
    # Add other settings as needed

    class Config:
        env_prefix = "MLO_"  # Environment variables will be prefixed with MLO_

settings = Settings()