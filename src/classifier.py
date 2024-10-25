# classifier.py
import logging
import time
from typing import Optional, Dict, Any, Tuple
import requests
import json

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
                "Could not connect to Ollama. Ensure Llama2 is installed with 'ollama pull llama2'. "
                f"Error: {str(e)}"
            )

    def _generate_classification_prompt(self, text: str) -> str:
        """Generate a prompt for category classification using Llama2's format"""
        categories = [
            f"{i+1}. {cat.value}" 
            for i, cat in enumerate(NewsCategory)
        ]
        categories_list = "\n".join(categories)
        
        return f"""<s>[INST] You are a financial news classifier. Analyze this article and classify it into exactly ONE category.

Available categories:
{categories_list}

Rules:
1. Read the article carefully and consider the main topic
2. Choose the MOST relevant category
3. Provide your response in JSON format:
   {{"category_number": X, "confidence": Y}}
   where X is the category number (1-9) and Y is your confidence (0-1)

Article:
{text}

Classify this article: [/INST]</s>"""

    def _generate_sentiment_prompt(self, text: str) -> str:
        """Generate a prompt for sentiment analysis using Llama2's format"""
        return f"""<s>[INST] You are a financial sentiment analyzer. Analyze the sentiment of this article.

Options:
1. positive (indicates growth, profit, success, or positive market outlook)
2. negative (indicates decline, loss, failure, or negative market outlook)
3. neutral (balanced or purely factual information)

Rules:
1. Consider the overall financial impact and market implications
2. Analyze the tone and factual content
3. Provide your response in JSON format:
   {{"sentiment_number": X, "confidence": Y}}
   where X is the sentiment number (1-3) and Y is your confidence (0-1)

Article:
{text}

Analyze the sentiment: [/INST]</s>"""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback"""
        try:
            # Find JSON-like content in the response
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, response)
            if matches:
                return json.loads(matches[0])
        except Exception as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
        return {}

    def _normalize_category(self, response: str) -> Tuple[str, float]:
        """Normalize category response and extract confidence"""
        try:
            # Parse JSON response
            result = self._parse_json_response(response)
            category_num = result.get('category_number')
            confidence = float(result.get('confidence', 0))
            
            # Number-based category selection
            if category_num:
                categories = list(NewsCategory)
                if 1 <= category_num <= len(categories):
                    return categories[category_num-1].value, confidence

            # Fallback to text analysis
            response = response.lower().strip()
            category_mapping = {
                'stock': ('stock_market', 0.8),
                'equity': ('stock_market', 0.8),
                'oil': ('oil_and_gas', 0.8),
                'gas': ('oil_and_gas', 0.8),
                'bank': ('banking', 0.8),
                'crypto': ('cryptocurrency', 0.8),
                'forex': ('forex', 0.8),
                'commodity': ('commodities', 0.8),
                'agriculture': ('agriculture', 0.8),
                'housing': ('housing', 0.8)
            }

            for key, (category, conf) in category_mapping.items():
                if key in response:
                    return category, conf

            return NewsCategory.OTHERS.value, 0.5

        except Exception as e:
            logger.warning(f"Category normalization error: {str(e)}")
            return NewsCategory.OTHERS.value, 0.0

    def _normalize_sentiment(self, response: str) -> Tuple[str, float]:
        """Normalize sentiment response and extract confidence"""
        try:
            # Parse JSON response
            result = self._parse_json_response(response)
            sentiment_num = result.get('sentiment_number')
            confidence = float(result.get('confidence', 0))
            
            # Number-based sentiment selection
            if sentiment_num:
                if sentiment_num == 1:
                    return SentimentType.POSITIVE.value, confidence
                elif sentiment_num == 2:
                    return SentimentType.NEGATIVE.value, confidence
                else:
                    return SentimentType.NEUTRAL.value, confidence

            # Fallback to text analysis
            response = response.lower().strip()
            if any(word in response for word in ['positive', 'growth', 'profit', 'success']):
                return SentimentType.POSITIVE.value, 0.8
            elif any(word in response for word in ['negative', 'decline', 'loss', 'fail']):
                return SentimentType.NEGATIVE.value, 0.8
            
            return SentimentType.NEUTRAL.value, 0.6

        except Exception as e:
            logger.warning(f"Sentiment normalization error: {str(e)}")
            return SentimentType.NEUTRAL.value, 0.0

    def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                    "max_tokens": config.MAX_TOKENS
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
                    processing_time=0.0,
                    confidence_score=0.0
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
                
                category, cat_confidence = self._normalize_category(category_raw)
                sentiment, sent_confidence = self._normalize_sentiment(sentiment_raw)
                
                # Average confidence score
                confidence_score = (cat_confidence + sent_confidence) / 2
                
                raw_response = (f"Category: {category_raw}, Confidence: {cat_confidence:.2f}\n"
                              f"Sentiment: {sentiment_raw}, Confidence: {sent_confidence:.2f}")
                success = confidence_score >= config.CONFIDENCE_THRESHOLD
            else:
                category = NewsCategory.OTHERS.value
                sentiment = SentimentType.NEUTRAL.value
                confidence_score = 0.0
                raw_response = None
                success = False

            processing_time = time.time() - start_time
            
            return NewsAnalysis(
                category=category,
                sentiment=sentiment,
                success=success,
                raw_response=raw_response,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {str(e)}")
            return NewsAnalysis(
                category=NewsCategory.OTHERS.value,
                sentiment=SentimentType.NEUTRAL.value,
                success=False,
                raw_response=str(e),
                processing_time=processing_time,
                confidence_score=0.0
            )