from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class NewsCategory(str, Enum):
    OIL_AND_GAS = "oil_and_gas"
    AGRICULTURE = "agriculture"
    HOUSING = "housing"
    BANKING = "banking"
    STOCK_MARKET = "stock_market"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITIES = "commodities"
    OTHERS = "others"  # Added new category

class NewsClassification(BaseModel):
    category: NewsCategory
    success: bool
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None