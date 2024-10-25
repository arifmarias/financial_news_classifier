# Financial News Classifier

A Python project that uses Ollama with the TinyLlama model to classify financial news articles into predefined categories. The system processes news articles from CSV files and classifies them into categories like oil and gas, agriculture, banking, cryptocurrency, etc.

[X] Author: Mohammed Arif

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Detailed Implementation](#detailed-implementation)
   - [models.py](#modelspy---data-structures)
   - [config.py](#configpy---configuration-management)
   - [classifier.py](#classifierpy---core-classification-logic)
   - [processor.py](#processorpy---batch-processing)
5. [Usage](#usage)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)
8. [Contributing](#contributing)
9. [License](#license)

## Requirements

- Python 3.8+
- Ollama installed and running
- TinyLlama model pulled in Ollama
- Required Python packages:
  ```
  pandas>=1.3.0
  pydantic>=2.0.0
  requests>=2.25.0
  tqdm>=4.65.0
  python-dotenv>=0.19.0
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-news-classifier.git
   cd financial-news-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama server:
   ```bash
   ollama serve
   ```

5. Pull the TinyLlama model:
   ```bash
   ollama pull tinyllama
   ```

6. Create necessary directories:
   ```bash
   mkdir -p data logs
   ```

## Project Structure

```
financial_news_classifier/
├── data/                  # Directory for input/output CSV files
│   ├── news_articles.csv  # Input news articles
│   └── processed_*.csv    # Processed output files
├── logs/                  # Log files directory
├── tests/                 # Test files
├── requirements.txt       # Project dependencies
└── src/
    ├── __init__.py
    ├── config.py         # Configuration settings
    ├── models.py         # Data models and enums
    ├── classifier.py     # Core classification logic
    └── processor.py      # CSV processing logic
```

## Detailed Implementation

### models.py - Data Structures

Defines the core data structures using Pydantic and Enums.

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

class NewsClassification(BaseModel):
    category: NewsCategory
    success: bool
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
```

Key Features:
- String-based enumeration for categories
- Pydantic model for validation
- Optional fields for metadata
- Type safety and validation

### config.py - Configuration Management

Centralized configuration using Pydantic BaseModel.

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
    
    # CSV settings
    CSV_INPUT_COLUMNS: list = ["Headline", "Date", "Article"]
    CSV_DATE_FORMAT: str = "%Y-%m-%d"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
```

Features:
- Environment variable support
- Type validation
- Automatic directory creation
- Configurable API parameters

### classifier.py - Core Classification Logic

Handles interaction with Ollama API and text classification.

#### Key Components:

1. **Initialization and Connection Verification**:
```python
class FinancialNewsClassifier:
    def __init__(self):
        self.api_url = config.OLLAMA_URL
        self.model_name = config.MODEL_NAME
        self._verify_ollama_connection()
```
- Verifies Ollama availability
- Configures API endpoint
- Initializes connection

2. **Prompt Engineering**:
```python
def _generate_prompt(self, text: str) -> str:
    """Generate structured classification prompt"""
    categories = [f"{i+1}. {cat.value}" 
                 for i, cat in enumerate(NewsCategory)]
    # Returns formatted prompt with instructions
```
- Creates numbered category list
- Clear classification instructions
- Structured format for consistency

3. **Category Normalization**:
```python
def _normalize_category(self, response: str) -> str:
    """Normalize model response to standard category"""
    # Multiple normalization strategies:
    # 1. Number extraction
    # 2. Text matching
    # 3. Keyword mapping
    # 4. Fallback handling
```

4. **API Interaction**:
```python
def _call_ollama(self, news_text: str) -> Optional[Dict[str, Any]]:
    """Call Ollama API with retry logic"""
    # Implements retry mechanism
    # Handles timeouts and errors
    # Returns processed response
```

5. **Classification Pipeline**:
```python
def classify_news(self, news_text: str) -> NewsClassification:
    """Classify a financial news article"""
    # 1. Input validation
    # 2. API call
    # 3. Response processing
    # 4. Result formatting
```

Features:
- Robust error handling
- Retry mechanisms
- Response validation
- Performance tracking

### processor.py - Batch Processing

Handles batch processing of news articles with progress tracking.

#### Key Components:

1. **CSV Validation**:
```python
def validate_csv(self, df: pd.DataFrame) -> bool:
    """Validate CSV structure"""
    required_columns = set(config.CSV_INPUT_COLUMNS)
    current_columns = set(df.columns)
    return required_columns.issubset(current_columns)
```

2. **DataFrame Processing**:
```python
def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Process articles in DataFrame"""
    # Progress tracking
    # Article processing
    # Error handling
    # Rate limiting
```

3. **File Processing**:
```python
def process_csv_file(self, input_file: Optional[Path] = None,
                    output_file: Optional[Path] = None) -> bool:
    """Process CSV file with news articles"""
    # File handling
    # CSV validation
    # Article processing
    # Results saving
```

4. **Statistics Generation**:
```python
def _log_statistics(self, df: pd.DataFrame):
    """Generate processing statistics"""
    # Success rates
    # Category distribution
    # Error analysis
```

Features:
- Progress tracking with tqdm
- Comprehensive error handling
- Statistical analysis
- Rate limiting
- Detailed logging

## Usage

1. Basic usage:
```python
from src.processor import NewsProcessor

# Initialize processor
processor = NewsProcessor()

# Process CSV file
success = processor.process_csv_file()
```

2. Custom file paths:
```python
from pathlib import Path

input_path = Path("data/custom_input.csv")
output_path = Path("data/custom_output.csv")

processor.process_csv_file(input_path, output_path)
```

3. Monitor progress:
```python
# Processing will show a progress bar:
Processing articles: 100%|██████████| 1000/1000 [00:30<00:00, 33.33 articles/s]
```

## Error Handling

1. **API Connection**:
   - Connection timeouts
   - Service unavailability
   - Response validation
   - Retry mechanisms

2. **Data Processing**:
   - Invalid CSV format
   - Missing columns
   - Malformed articles
   - Category parsing errors

3. **Resource Management**:
   - Memory monitoring
   - Processing timeouts
   - Rate limiting

## Best Practices

1. **Logging**:
   - Structured logging
   - Error tracking
   - Performance monitoring
   - Statistics collection

2. **Configuration**:
   - Environment variables
   - Type validation
   - Centralized settings
   - Documentation

3. **Error Recovery**:
   - Automatic retries
   - Progress preservation
   - Detailed error reporting
   - Fallback mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For more information or support, please open an issue in the GitHub repository.