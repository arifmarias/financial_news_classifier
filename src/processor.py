import logging
import time
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

from .classifier import FinancialNewsClassifier
from .models import NewsClassification
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
        """Process articles in a DataFrame"""
        try:
            # Create a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Initialize category column
            processed_df['Category'] = None
            
            total_rows = len(processed_df)
            logger.info(f"Starting to process {total_rows} articles")
            
            # Process each article with progress bar
            for idx in tqdm(range(total_rows), desc="Processing articles"):
                try:
                    article = processed_df.iloc[idx]['Article']
                    # Skip empty articles
                    if pd.isna(article) or not article.strip():
                        logger.warning(f"Empty article at index {idx}")
                        processed_df.at[idx, 'Category'] = 'UNKNOWN'
                        continue
                        
                    # Classify the article
                    result = self.classifier.classify_news(str(article))
                    processed_df.at[idx, 'Category'] = result.category
                    
                    # Add small delay to prevent overwhelming Ollama
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing article at index {idx}: {str(e)}")
                    processed_df.at[idx, 'Category'] = 'ERROR'
            
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
            # Use default paths if not provided
            input_file = input_file or config.INPUT_CSV
            output_file = output_file or config.OUTPUT_CSV
            
            logger.info(f"Reading CSV file: {input_file}")
            df = pd.read_csv(input_file)
            
            # Validate CSV structure
            if not self.validate_csv(df):
                raise ValueError("Invalid CSV structure")
            
            # Process the articles
            processed_df = self.process_dataframe(df)
            
            # Save the results
            processed_df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to: {output_file}")
            
            # Log processing statistics
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
            
            stats = {
                'Total articles': total_articles,
                'Successfully categorized': categorized,
                'Unknown articles': unknown,
                'Errors': errors,
                'Success rate': f"{(categorized/total_articles)*100:.2f}%"
            }
            
            # Category distribution
            category_dist = df['Category'].value_counts().to_dict()
            
            logger.info("Processing Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            
            logger.info("Category Distribution:")
            for category, count in category_dist.items():
                logger.info(f"{category}: {count} ({(count/total_articles)*100:.2f}%)")
                
        except Exception as e:
            logger.error(f"Failed to log statistics: {str(e)}")