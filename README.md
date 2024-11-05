# Financial News Classifier with Llama2

A comprehensive Python application that uses the Llama2 model through Ollama to classify financial news articles and perform sentiment analysis.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Detailed Component Explanation](#detailed-component-explanation)
5. [Usage](#usage)
6. [Flow of Execution](#flow-of-execution)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

## Overview

This project classifies financial news articles into predefined categories and analyzes their sentiment using the Llama2 language model. It processes CSV files containing news articles and generates detailed analysis with confidence scores.

### Key Features
- Categorizes news articles into 9 financial sectors
- Performs sentiment analysis (positive/negative/neutral)
- Provides confidence scores for predictions
- Generates detailed statistics and logs
- Handles batch processing with progress tracking

## Project Structure

```
financial_news_classifier/
├── src/
│   ├── __init__.py
│   ├── models.py        # Data models and enums
│   ├── config.py        # Configuration settings
│   ├── classifier.py    # Core classification logic
│   └── processor.py     # CSV processing logic
├── data/                # Input/output CSV files
├── logs/               # Log files
├── requirements.txt
├── README.md
└── main.py            # Entry point
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Ollama:
- Visit [Ollama's website](https://ollama.ai) for installation
- Or use command line: `curl https://ollama.ai/install.sh | sh`

3. Install Llama2 model:
```bash
ollama pull llama2
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Detailed Component Explanation

### 1. models.py
Defines the core data structures using Pydantic and Enums.

Key Components:
- `NewsCategory`: Enum for financial sectors
  - oil_and_gas, agriculture, housing, etc.
- `SentimentType`: Enum for sentiment values
  - positive, negative, neutral
- `NewsAnalysis`: Pydantic model for results
  - Includes category, sentiment, confidence scores

### 2. config.py
Manages all configuration settings using Pydantic.

Key Settings:
- Ollama API configuration
  - URL, model name, timeout settings
- Processing parameters
  - Batch size, temperature, confidence threshold
- File paths and CSV settings

### 3. classifier.py
Core classification logic using Llama2.

Key Functions:
1. `_verify_ollama_connection()`
   - Checks if Ollama is running
   - Validates API accessibility

2. `_generate_classification_prompt()`
   - Creates structured prompts for category classification
   - Uses Llama2's instruction format

3. `_generate_sentiment_prompt()`
   - Creates prompts for sentiment analysis
   - Includes clear guidelines for model

4. `_normalize_category()` and `_normalize_sentiment()`
   - Process model responses
   - Extract categories and confidence scores
   - Include fallback mechanisms

5. `analyze_news()`
   - Main analysis function
   - Combines category and sentiment analysis
   - Handles errors and timeouts

### 4. processor.py
Handles batch processing of news articles.

Key Functions:
1. `validate_csv()`
   - Checks required columns
   - Validates input format

2. `process_dataframe()`
   - Processes articles in batches
   - Shows progress bar
   - Handles rate limiting

3. `process_csv_file()`
   - Manages file I/O
   - Coordinates processing
   - Generates statistics

4. `_log_statistics()`
   - Calculates success rates
   - Generates distribution reports
   - Logs detailed metrics

## Flow of Execution

1. **Initialization**:
   - `main.py` creates directories
   - Sets up logging
   - Initializes processor

2. **Data Loading**:
   - Reads input CSV
   - Validates structure
   - Creates processing pipeline

3. **Processing**:
   - For each article:
     1. Category classification
     2. Sentiment analysis
     3. Confidence calculation
     4. Result storage

4. **Output**:
   - Saves processed data
   - Generates statistics
   - Creates detailed logs

## Usage

1. Prepare input CSV with columns:
   - Headline
   - Date
   - Article

2. Run the classifier:
```bash
python main.py
```

3. Check outputs in:
   - `data/` for processed CSV
   - `logs/` for processing logs

## Customization

1. Modify Categories:
   - Edit `NewsCategory` in models.py
   - Update prompts in classifier.py

2. Adjust Parameters:
   - Edit config.py for:
     - Model parameters
     - Processing settings
     - Confidence thresholds

3. Custom Processing:
   - Modify processor.py for:
     - Different input formats
     - Additional analytics
     - Custom statistics

## Troubleshooting

1. Import Errors:
   - Verify project structure
   - Check virtual environment
   - Confirm __init__.py exists

2. Ollama Issues:
   - Verify Ollama is running
   - Check model installation
   - Confirm API accessibility

3. Processing Errors:
   - Check CSV format
   - Verify column names
   - Monitor memory usage

## Common Issues and Solutions

1. "ImportError":
   - Run from project root
   - Check file structure
   - Verify imports

2. "OllamaConnectionError":
   - Start Ollama service
   - Check API URL
   - Verify model installation

3. "CSV Validation Error":
   - Check column names
   - Verify data format
   - Confirm file encoding