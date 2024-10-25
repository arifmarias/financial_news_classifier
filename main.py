import logging
from datetime import datetime
from pathlib import Path
from src.config import config
from src.processor import NewsProcessor

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"processing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def main():
    # Create necessary directories
    config.create_directories()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting financial news classification process")
    
    try:
        # Initialize processor
        processor = NewsProcessor()
        
        # Process the CSV file
        success = processor.process_csv_file()
        
        if success:
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing completed with errors")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()