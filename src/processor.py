# processor.py
import logging
import time
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

from .classifier import FinancialNewsClassifier
from .models import NewsAnalysis
from .config import config

logger = logging.getLogger(__name__)

class NewsProcessor:
    def __init__(self):
        self.classifier = FinancialNewsClassifier()

    def validate_csv(self, df: pd.DataFrame) -> bool:
        """Validate if CSV has required columns"""
        required_columns = set(config.CSV_INPUT_COLUMNS)
        current_columns = set(df.columns)
        if not required_columns.issubset(current_columns):
            missing = required_columns - current_columns
            logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process articles in DataFrame"""
        try:
            processed_df = df.copy()
            processed_df['Category'] = None
            processed_df['Sentiment'] = None
            processed_df['Confidence'] = None
            
            total_rows = len(processed_df)
            logger.info(f"Starting to process {total_rows} articles")
            
            for idx in tqdm(range(total_rows), desc="Processing articles"):
                try:
                    article = processed_df.iloc[idx]['Article']
                    if pd.isna(article) or not article.strip():
                        logger.warning(f"Empty article at index {idx}")
                        processed_df.at[idx, 'Category'] = 'UNKNOWN'
                        processed_df.at[idx, 'Sentiment'] = 'NEUTRAL'
                        processed_df.at[idx, 'Confidence'] = 0.0
                        continue
                        
                    result = self.classifier.analyze_news(str(article))
                    processed_df.at[idx, 'Category'] = result.category
                    processed_df.at[idx, 'Sentiment'] = result.sentiment
                    processed_df.at[idx, 'Confidence'] = result.confidence_score
                    
                    time.sleep(1.0)  # Rate limiting for Llama2
                    
                except Exception as e:
                    logger.error(f"Error processing article at index {idx}: {str(e)}")
                    processed_df.at[idx, 'Category'] = 'ERROR'
                    processed_df.at[idx, 'Sentiment'] = 'NEUTRAL'
                    processed_df.at[idx, 'Confidence'] = 0.0
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Failed to process DataFrame: {str(e)}")
            raise

    def process_csv_file(
        self,
        input_file: Optional[Path] = None,
        output_file: Optional[Path] = None
    ) -> bool:
        """Process a CSV file containing news articles"""
        try:
            input_file = input_file or config.INPUT_CSV
            output_file = output_file or config.OUTPUT_CSV
            
            logger.info(f"Reading CSV file: {input_file}")
            df = pd.read_csv(input_file)
            
            if not self.validate_csv(df):
                raise ValueError("Invalid CSV structure")
            
            processed_df = self.process_dataframe(df)
            
            processed_df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to: {output_file}")
            
            self._log_statistics(processed_df)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {str(e)}")
            return False

    def _log_statistics(self, df: pd.DataFrame):
        """Log processing statistics"""
        try:
            total_articles = len(df)
            categorized = df['Category'].notna().sum()
            unknown = (df['Category'] == 'UNKNOWN').sum()
            errors = (df['Category'] == 'ERROR').sum()
            
            # Basic statistics
            stats = {
                'Total articles': total_articles,
                'Successfully categorized': categorized,
                'Unknown articles': unknown,
                'Errors': errors,
                'Success rate': f"{(categorized/total_articles)*100:.2f}%"
            }
            
            # Confidence statistics
            if 'Confidence' in df.columns:
                confidence_stats = {
                    'Average Confidence': f"{df['Confidence'].mean():.2f}",
                    'High Confidence (>0.8)': (df['Confidence'] > 0.8).sum(),
                    'Low Confidence (<0.5)': (df['Confidence'] < 0.5).sum()
                }
                stats.update(confidence_stats)
            
            # Category distribution
            category_dist = df['Category'].value_counts().to_dict()
            sentiment_dist = df['Sentiment'].value_counts().to_dict()
            
            # Log basic statistics
            logger.info("\nProcessing Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            
            # Log category distribution
            logger.info("\nCategory Distribution:")
            for category, count in category_dist.items():
                percentage = (count/total_articles)*100
                logger.info(f"{category}: {count} ({percentage:.2f}%)")
                
            # Log sentiment distribution
            logger.info("\nSentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count/total_articles)*100
                logger.info(f"{sentiment}: {count} ({percentage:.2f}%)")
            
            # Log category-sentiment combinations
            logger.info("\nCategory-Sentiment Distribution:")
            for category in df['Category'].unique():
                category_data = df[df['Category'] == category]
                logger.info(f"\n{category}:")
                sentiment_counts = category_data['Sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    percentage = (count/len(category_data))*100
                    logger.info(f"  {sentiment}: {count} ({percentage:.2f}%)")
                
        except Exception as e:
            logger.error(f"Failed to log statistics: {str(e)}")