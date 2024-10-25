# Financial News Classifier

A Python project that uses Ollama with the TinyLlama model to classify financial news articles into categories. The system processes news articles and classifies them into categories like oil and gas, agriculture, banking, cryptocurrency, etc.

[X] Author: Mohammed Arif
## Requirements

- Python 3.8+
- Ollama installed and running
- TinyLlama model pulled in Ollama

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
5. Pull the TinyLlama model:
   ```bash
   ollama pull tinyllama
   ```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Process sample articles
2. Save results to the data directory
3. Print classification results
4. Save logs to the logs directory

## Project Structure

- `src/`: Source code
  - `classifier.py`: Main classification logic
  - `processor.py`: Batch processing functionality
  - `config.py`: Configuration settings
  - `models.py`: Data models
- `tests/`: Test files
- `data/`: Input/output data
- `logs/`: Log files

## Detailed Code Documentation

### models.py

The models module defines the core data structures used throughout the application.

#### NewsCategory

```python
class NewsCategory(str, Enum):
    OIL_AND_GAS = "oil_and_gas"
    AGRICULTURE = "agriculture"
    HOUSING = "housing"
    BANKING = "banking"
    STOCK_MARKET = "stock_market"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITIES = "commodities"
    OTHERS = "others"
```

This enum defines all possible categories for news classification. Using an enum ensures type safety and prevents invalid categories.

#### NewsClassification

```python
class NewsClassification(BaseModel):
    category: NewsCategory
    success: bool
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
```

A Pydantic model that represents the classification result, including:
- The assigned category
- Success status
- Raw model response
- Processing time

### config.py

Configuration management using Pydantic for type safety and validation.

```python
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
    
    # CSV settings and paths
    CSV_INPUT_COLUMNS: list = ["Headline", "Date", "Article"]
    CSV_DATE_FORMAT: str = "%Y-%m-%d"
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
```

Key configurations include:
- Ollama API settings
- Request handling parameters
- Model configuration
- File paths and formats

### classifier.py

The core classification logic that interacts with the Ollama API.

#### Key Components:

1. Initialization and Connection Verification
```python
def __init__(self):
    self.api_url = config.OLLAMA_URL
    self.model_name = config.MODEL_NAME
    self._verify_ollama_connection()
```

2. Prompt Generation
```python
def _generate_prompt(self, text: str) -> str:
    """Generate a structured prompt for the model"""
    categories = [f"{i+1}. {cat.value}" for i, cat in enumerate(NewsCategory)]
    # Returns formatted prompt with categories and instructions
```

3. Category Normalization
```python
def _normalize_category(self, response: str) -> str:
    """Normalize model response to standard categories"""
    # Handles numerical and text-based responses
    # Maps common terms to categories
    # Includes fallback mechanisms
```

4. API Interaction
```python
def _call_ollama(self, news_text: str) -> Optional[Dict[str, Any]]:
    """Handle API calls with retry logic"""
    # Implements retry mechanism with exponential backoff
    # Handles timeouts and errors
    # Returns processed response
```

### processor.py

Handles batch processing of news articles from CSV files.

#### Key Features:

1. CSV Validation
```python
def validate_csv(self, df: pd.DataFrame) -> bool:
    """Validate CSV structure"""
    required_columns = set(config.CSV_INPUT_COLUMNS)
    # Checks for required columns
```

2. DataFrame Processing
```python
def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Process articles in batches"""
    # Includes progress tracking
    # Handles individual article failures
    # Implements rate limiting
```

3. Statistics Logging
```python
def _log_statistics(self, df: pd.DataFrame):
    """Log processing statistics"""
    # Calculates success rates
    # Logs category distribution
    # Provides processing summary
```

## Error Handling

The system implements comprehensive error handling:

1. API Connection:
   - Connection timeouts
   - Service unavailability
   - Response validation

2. Data Processing:
   - Invalid CSV format
   - Missing/malformed articles
   - Category normalization errors

3. Resource Management:
   - Memory usage monitoring
   - Processing time tracking
   - API rate limiting

## Best Practices

1. Logging:
   - Structured logging throughout
   - Processing metrics tracking
   - Error pattern monitoring

2. Configuration:
   - Environment variable support
   - Type validation
   - Documented settings

3. Error Recovery:
   - Retry mechanisms
   - Progress preservation
   - Detailed error reporting
