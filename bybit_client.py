import logging
from pybit.unified_trading import HTTP
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the Bybit HTTP session.
        """
        try:
            self.session = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret,
            )
            logger.info(f"Bybit Client initialized (Testnet: {testnet})")
        except Exception as e:
            logger.error(f"Failed to initialize Bybit Client: {e}")
            raise

    def get_instruments_info(self, category: str, symbol: str) -> Dict:
        """
        Get instrument information (tick size, lot size, etc).
        """
        try:
            response = self.session.get_instruments_info(
                category=category,
                symbol=symbol,
            )
            if response['retCode'] == 0:
                result = response['result']['list'][0]
                return result
            else:
                logger.error(f"Error fetching instrument info: {response}")
                return {}
        except Exception as e:
            logger.error(f"Exception in get_instruments_info: {e}")
            return {}

    def set_leverage(self, category: str, symbol: str, leverage: str):
        """
        Set the leverage for a symbol.
        """
        try:
            # First check current leverage to avoid error if already set
            # Ideally we just try to set it, and ignore "Position not modified" errors
            self.session.set_leverage(
                category=category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.info(f"Leverage set to {leverage} for {symbol}")
        except Exception as e:
            # 110043 is the error code for "leverage not modified" (already set)
            if "110043" in str(e): 
                logger.info(f"Leverage already set to {leverage}")
            else:
                logger.error(f"Failed to set leverage: {e}")

    def get_klines(self, category: str, symbol: str, interval: int, limit: int = 200) -> List[Dict]:
        """
        Get historical klines (candlesticks).
        Returns a list of raw kline data.
        """
        try:
            response = self.session.get_kline(
                category=category,
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            if response['retCode'] == 0:
                # Bybit returns klines in reverse chronological order (newest first). 
                # We often usually want them sorted by time ascending for TA libs.
                # The raw list is [startTime, open, high, low, close, volume, turnover]
                return response['result']['list']
            else:
                logger.error(f"Error fetching klines: {response}")
                return []
        except Exception as e:
            logger.error(f"Exception in get_klines: {e}")
            return []

    def get_tickers(self, category: str, symbol: str) -> Dict:
        """
        Get latest ticker info (best bid/ask, last price).
        """
        try:
            response = self.session.get_tickers(
                category=category,
                symbol=symbol,
            )
            if response['retCode'] == 0:
                return response['result']['list'][0]
            else:
                logger.error(f"Error fetching ticker: {response}")
                return {}
        except Exception as e:
            logger.error(f"Exception in get_tickers: {e}")
            return {}

    def get_open_positions(self, category: str, symbol: str) -> List[Dict]:
        """
        Get current open positions.
        """
        try:
            response = self.session.get_positions(
                category=category,
                symbol=symbol,
            )
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                logger.error(f"Error fetching positions: {response}")
                return []
        except Exception as e:
            logger.error(f"Exception in get_open_positions: {e}")
            return []

    def place_order(self, category: str, symbol: str, side: str, qty: str, 
                   order_type: str = "Market", price: Optional[str] = None, 
                   take_profit: Optional[str] = None, stop_loss: Optional[str] = None):
        """
        Place a new order with optional TP/SL.
        """
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
        }
        if price:
            params["price"] = price
        if take_profit:
            params["takeProfit"] = take_profit
        if stop_loss:
            params["stopLoss"] = stop_loss

        try:
            response = self.session.place_order(**params)
            if response['retCode'] == 0:
                logger.info(f"Order placed successfully: {response['result']}")
                return response['result']
            else:
                logger.error(f"Error placing order: {response}")
                return None
        except Exception as e:
            logger.error(f"Exception in place_order: {e}")
            return None
