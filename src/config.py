# config.py
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

class Config(BaseModel):
    # Ollama settings
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    MODEL_NAME: str = "tinyllama"
    
    # API settings
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    
    # Processing settings
    BATCH_SIZE: int = 10
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    
    # CSV settings
    CSV_INPUT_COLUMNS: list = ["Headline", "Date", "Article"]
    CSV_DATE_FORMAT: str = "%Y-%m-%d"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    INPUT_CSV: Path = DATA_DIR / "news_articles.csv"
    OUTPUT_CSV: Path = DATA_DIR / f"processed_articles_{datetime.now():%Y%m%d_%H%M%S}.csv"
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

config = Config()