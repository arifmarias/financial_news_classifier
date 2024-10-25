# classifier.py
import logging
import time
from typing import Optional, Dict, Any
import requests

from .config import config
from .models import NewsCategory, SentimentType, NewsAnalysis

logger = logging.getLogger(__name__)

class OllamaConnectionError(Exception):
    """Raised when there are issues connecting to Ollama"""
    pass

class FinancialNewsClassifier:
    def __init__(self):
        self.api_url = config.OLLAMA_URL
        self.model_name = config.MODEL_NAME
        self._verify_ollama_connection()
        logger.info(f"Initialized classifier with model: {self.model_name}")

    def _verify_ollama_connection(self) -> None:
        """Verify that Ollama is running and accessible"""
        try:
            response = requests.get(
                self.api_url.replace("/generate", "/version"),
                timeout=5
            )
            response.raise_for_status()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            raise OllamaConnectionError(
                "Could not connect to Ollama. Please ensure Ollama is running. "
                f"Error: {str(e)}"
            )

    def _generate_classification_prompt(self, text: str) -> str:
        """Generate a prompt for category classification"""
        categories = [
            f"{i+1}. {cat.value}" 
            for i, cat in enumerate(NewsCategory)
        ]
        categories_list = "\n".join(categories)
        
        return f"""Classify this financial news article into ONE of these categories:

{categories_list}

Rules:
1. Choose ONLY ONE category number
2. If the article doesn't clearly fit into specific categories 1-8, choose 9 (others)
3. Respond ONLY with the category number (1-9)
4. Don't explain your choice, just provide the number

Article:
{text}

Category number:"""

    def _generate_sentiment_prompt(self, text: str) -> str:
        """Generate a prompt for sentiment analysis"""
        return f"""Analyze the sentiment of this financial news article. Choose ONE:
1. positive (indicates growth, profit, success, or positive market outlook)
2. negative (indicates decline, loss, failure, or negative market outlook)
3. neutral (balanced or purely factual information)

Rules:
1. Consider the overall financial impact and market implications
2. Respond ONLY with the number (1-3)
3. Don't explain your choice

Article:
{text}

Sentiment number:"""

    def _normalize_category(self, response: str) -> str:
        """Normalize category response"""
        try:
            # First try to extract numbers
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                number = int(numbers[0])
                categories = list(NewsCategory)
                if 1 <= number <= len(categories):
                    return categories[number-1].value

            # Text-based matching fallback
            response = response.lower().strip()
            response = ''.join(c for c in response if c.isalnum() or c in ['_', ' '])
            response = response.replace(' ', '_')

            # Category mapping
            category_mapping = {
                'stock': 'stock_market',
                'equity': 'stock_market',
                'shares': 'stock_market',
                'market': 'stock_market',
                'oil': 'oil_and_gas',
                'gas': 'oil_and_gas',
                'energy': 'oil_and_gas',
                'bank': 'banking',
                'loan': 'banking',
                'crypto': 'cryptocurrency',
                'bitcoin': 'cryptocurrency',
                'forex': 'forex',
                'currency': 'forex',
                'commodity': 'commodities'
            }

            for key, value in category_mapping.items():
                if key in response:
                    return value

            return NewsCategory.OTHERS.value

        except Exception as e:
            logger.warning(f"Category normalization error: {str(e)}")
            return NewsCategory.OTHERS.value

    def _normalize_sentiment(self, response: str) -> str:
        """Normalize sentiment response"""
        try:
            # Extract numbers
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                number = int(numbers[0])
                if number == 1:
                    return SentimentType.POSITIVE.value
                elif number == 2:
                    return SentimentType.NEGATIVE.value
                else:
                    return SentimentType.NEUTRAL.value

            # Text matching fallback
            response = response.lower().strip()
            if any(word in response for word in ['positive', 'growth', 'profit', 'success', 'up']):
                return SentimentType.POSITIVE.value
            elif any(word in response for word in ['negative', 'decline', 'loss', 'down', 'fail']):
                return SentimentType.NEGATIVE.value
            
            return SentimentType.NEUTRAL.value

        except Exception as e:
            logger.warning(f"Sentiment normalization error: {str(e)}")
            return SentimentType.NEUTRAL.value

    def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P
                }
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == config.MAX_RETRIES - 1:
                    logger.error(f"Failed after {config.MAX_RETRIES} attempts")
                    return None
                time.sleep(config.RETRY_DELAY * (2 ** attempt))
        
        return None

    def analyze_news(self, news_text: str) -> NewsAnalysis:
        """Analyze news for category and sentiment"""
        start_time = time.time()
        try:
            # Preprocess the text
            news_text = news_text.strip()
            if not news_text:
                return NewsAnalysis(
                    category=NewsCategory.OTHERS.value,
                    sentiment=SentimentType.NEUTRAL.value,
                    success=False,
                    raw_response="Empty text",
                    processing_time=0.0
                )

            # Get category
            category_prompt = self._generate_classification_prompt(news_text)
            category_response = self._call_ollama(category_prompt)
            
            # Get sentiment
            sentiment_prompt = self._generate_sentiment_prompt(news_text)
            sentiment_response = self._call_ollama(sentiment_prompt)

            if category_response and sentiment_response:
                category_raw = category_response.get('response', '').strip()
                sentiment_raw = sentiment_response.get('response', '').strip()
                
                category = self._normalize_category(category_raw)
                sentiment = self._normalize_sentiment(sentiment_raw)
                
                raw_response = f"Category: {category_raw}, Sentiment: {sentiment_raw}"
                success = True
            else:
                category = NewsCategory.OTHERS.value
                sentiment = SentimentType.NEUTRAL.value
                raw_response = None
                success = False

            processing_time = time.time() - start_time
            
            return NewsAnalysis(
                category=category,
                sentiment=sentiment,
                success=success,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {str(e)}")
            return NewsAnalysis(
                category=NewsCategory.OTHERS.value,
                sentiment=SentimentType.NEUTRAL.value,
                success=False,
                raw_response=str(e),
                processing_time=processing_time
            )