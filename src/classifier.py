import logging
import time
from typing import Optional, Dict, Any
import requests

from .config import config
from .models import NewsCategory, NewsClassification

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
                "Could not connect to Ollama. Please ensure Ollama is running with 'ollama serve'. "
                f"Error: {str(e)}"
            )

    def _generate_prompt(self, text: str) -> str:
        """Generate a more structured and constrained prompt"""
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

    def _normalize_category(self, response: str) -> str:
        """Improved category normalization"""
        try:
            # First try to extract just numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                # Take the first number found
                number = int(numbers[0])
                # Map numbers to categories (1-based index)
                categories = list(NewsCategory)
                if 1 <= number <= len(categories):
                    return categories[number-1].value

            # If number parsing fails, try text matching
            response = response.lower().strip()
            response = ''.join(c for c in response if c.isalnum() or c in ['_', ' '])
            response = response.replace(' ', '_')

            # Direct match with category values
            valid_categories = [c.value for c in NewsCategory]
            if response in valid_categories:
                return response

            # Category mapping for common terms
            category_mapping = {
                'stock': 'stock_market',
                'equity': 'stock_market',
                'shares': 'stock_market',
                'market': 'stock_market',
                'oil': 'oil_and_gas',
                'gas': 'oil_and_gas',
                'energy': 'oil_and_gas',
                'petroleum': 'oil_and_gas',
                'crop': 'agriculture',
                'farm': 'agriculture',
                'grain': 'agriculture',
                'house': 'housing',
                'property': 'housing',
                'real_estate': 'housing',
                'mortgage': 'housing',
                'bank': 'banking',
                'loan': 'banking',
                'credit': 'banking',
                'crypto': 'cryptocurrency',
                'bitcoin': 'cryptocurrency',
                'ethereum': 'cryptocurrency',
                'currency': 'forex',
                'exchange_rate': 'forex',
                'commodity': 'commodities',
                'gold': 'commodities',
                'metal': 'commodities'
            }

            # Check for mapped terms
            for key, value in category_mapping.items():
                if key in response:
                    return value

            # If no match found, return OTHERS instead of stock_market
            logger.info(f"Could not match category '{response}', categorizing as 'others'")
            return NewsCategory.OTHERS.value

        except Exception as e:
            logger.warning(f"Error in category normalization: {str(e)}")
            return NewsCategory.OTHERS.value

    def _call_ollama(self, news_text: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API with improved error handling"""
        for attempt in range(config.MAX_RETRIES):
            try:
                prompt = self._generate_prompt(news_text)
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Lower temperature for more consistent results
                    "top_p": 0.9
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
                    logger.error(f"Failed to get response after {config.MAX_RETRIES} attempts")
                    return None
                time.sleep(config.RETRY_DELAY * (2 ** attempt))
        
        return None

    def classify_news(self, news_text: str) -> NewsClassification:
        """Classify a piece of financial news"""
        start_time = time.time()
        try:
            # Preprocess the news text
            news_text = news_text.strip()
            if not news_text:
                return NewsClassification(
                    category=NewsCategory.OTHERS.value,
                    success=False,
                    raw_response="Empty text",
                    processing_time=0.0
                )

            response = self._call_ollama(news_text)
            if response:
                raw_response = response.get('response', '').strip()
                category = self._normalize_category(raw_response)
                success = True
            else:
                category = NewsCategory.OTHERS.value
                raw_response = None
                success = False

            processing_time = time.time() - start_time
            
            return NewsClassification(
                category=category,
                success=success,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            processing_time = time.time() - start_time
            return NewsClassification(
                category=NewsCategory.OTHERS.value,
                success=False,
                raw_response=str(e),
                processing_time=processing_time
            )