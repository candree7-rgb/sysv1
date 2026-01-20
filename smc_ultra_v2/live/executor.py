"""
SMC Ultra V2 - Order Executor
=============================
Führt Orders auf Bybit aus.

Features:
- Limit und Market Orders
- Automatic TP/SL Placement
- Position Tracking
- Error Handling
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from pybit.unified_trading import HTTP

from config.settings import config
from strategy import Signal, Trade


class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: str = None
    filled_price: float = None
    filled_qty: float = None
    error: str = None
    timestamp: datetime = None


@dataclass
class Position:
    """Active position on exchange"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    take_profit: float = None
    stop_loss: float = None


class BybitExecutor:
    """
    Executes orders on Bybit.

    Handles:
    - Order placement
    - Position management
    - TP/SL management
    """

    def __init__(self):
        self.api_config = config.api
        self.client = HTTP(
            testnet=self.api_config.testnet,
            api_key=self.api_config.api_key,
            api_secret=self.api_config.api_secret
        )

        self.positions: Dict[str, Position] = {}

    def get_balance(self) -> Dict:
        """Get account balance"""
        def safe_float(val, default=0.0):
            """Safely convert to float, handling empty strings"""
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        try:
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )

            if response['retCode'] == 0:
                coins = response['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                if usdt:
                    equity = safe_float(usdt.get('equity'), 0)
                    wallet_balance = safe_float(usdt.get('walletBalance'), 0)
                    available_to_withdraw = safe_float(usdt.get('availableToWithdraw'), 0)
                    pnl = safe_float(usdt.get('unrealisedPnl'), 0)

                    # Use walletBalance for trading, not availableToWithdraw
                    available = wallet_balance if wallet_balance > 0 else equity

                    # Check if account has any balance
                    if equity == 0 and available == 0:
                        return {'error': 'No USDT balance in testnet account. Please add funds on testnet.bybit.com'}

                    return {
                        'equity': equity,
                        'available': available,
                        'wallet_balance': wallet_balance,
                        'unrealized_pnl': pnl
                    }

            return {'error': response.get('retMsg', 'Unknown error')}

        except Exception as e:
            return {'error': str(e)}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        try:
            response = self.client.get_positions(
                category="linear",
                symbol=symbol
            )

            if response['retCode'] == 0 and response['result']['list']:
                pos = response['result']['list'][0]
                size = float(pos['size'])

                if size > 0:
                    return Position(
                        symbol=symbol,
                        side=pos['side'],
                        size=size,
                        entry_price=float(pos['avgPrice']),
                        leverage=int(pos['leverage']),
                        unrealized_pnl=float(pos['unrealisedPnl']),
                        take_profit=float(pos['takeProfit']) if pos['takeProfit'] else None,
                        stop_loss=float(pos['stopLoss']) if pos['stopLoss'] else None
                    )

            return None

        except Exception as e:
            print(f"Error getting position: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        try:
            response = self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return response['retCode'] == 0

        except Exception as e:
            print(f"Error setting leverage: {e}")
            return False

    def open_position(
        self,
        signal: Signal,
        qty: float = None
    ) -> OrderResult:
        """
        Open a position based on signal using LIMIT order for precise entry.

        Args:
            signal: Trading signal with entry_price
            qty: Quantity (calculated from signal if not provided)

        Returns:
            OrderResult
        """
        try:
            symbol = signal.symbol
            side = OrderSide.BUY.value if signal.direction == 'long' else OrderSide.SELL.value

            # Set leverage
            self.set_leverage(symbol, signal.leverage)

            # Calculate quantity if not provided
            if qty is None:
                qty = self._calc_position_size(signal)

            if qty <= 0:
                return OrderResult(success=False, error="Invalid quantity")

            # Place LIMIT order at exact entry price for better RR
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                price=str(signal.entry_price),
                qty=str(qty),
                timeInForce="PostOnly",  # Maker only = lower fees
                reduceOnly=False,
                closeOnTrigger=False,
                # Attach TP/SL directly to order
                takeProfit=str(signal.take_profit),
                stopLoss=str(signal.stop_loss),
                tpslMode="Full",
                tpOrderType="Limit",  # TP as Limit order
                slOrderType="Market"  # SL as Market for guaranteed exit
            )

            if response['retCode'] != 0:
                return OrderResult(
                    success=False,
                    error=response['retMsg']
                )

            order_id = response['result']['orderId']

            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=signal.entry_price,
                filled_qty=qty,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            response = self.client.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            return response['retCode'] == 0
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open/pending orders"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol

            response = self.client.get_open_orders(**params)
            if response['retCode'] == 0:
                return response['result']['list']
            return []
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return []

    def close_position(
        self,
        symbol: str,
        reason: str = "manual"
    ) -> OrderResult:
        """Close an open position"""
        try:
            position = self.get_position(symbol)
            if not position:
                return OrderResult(success=False, error="No position found")

            # Opposite side to close
            side = OrderSide.SELL.value if position.side == "Buy" else OrderSide.BUY.value

            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType=OrderType.MARKET.value,
                qty=str(position.size),
                timeInForce="GTC",
                reduceOnly=True
            )

            if response['retCode'] != 0:
                return OrderResult(
                    success=False,
                    error=response['retMsg']
                )

            return OrderResult(
                success=True,
                order_id=response['result']['orderId'],
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def update_tp_sl(
        self,
        symbol: str,
        take_profit: float = None,
        stop_loss: float = None
    ) -> bool:
        """Update TP/SL for an open position"""
        try:
            position = self.get_position(symbol)
            if not position:
                return False

            params = {
                "category": "linear",
                "symbol": symbol,
                "tpslMode": "Full",
                "positionIdx": 0
            }

            if take_profit:
                params["takeProfit"] = str(take_profit)
            if stop_loss:
                params["stopLoss"] = str(stop_loss)

            response = self.client.set_trading_stop(**params)
            return response['retCode'] == 0

        except Exception as e:
            print(f"Error updating TP/SL: {e}")
            return False

    def _set_tp_sl(
        self,
        symbol: str,
        direction: str,
        take_profit: float,
        stop_loss: float
    ) -> bool:
        """Set TP/SL after opening position"""
        try:
            response = self.client.set_trading_stop(
                category="linear",
                symbol=symbol,
                takeProfit=str(take_profit),
                stopLoss=str(stop_loss),
                tpslMode="Full",
                positionIdx=0
            )
            return response['retCode'] == 0

        except Exception as e:
            print(f"Error setting TP/SL: {e}")
            return False

    def _calc_position_size(self, signal: Signal) -> float:
        """Calculate position size based on signal and risk"""
        balance = self.get_balance()
        if 'error' in balance:
            return 0

        equity = balance['available']

        # Risk amount
        risk_pct = config.risk.max_risk_per_trade_pct
        risk_usd = equity * (risk_pct / 100)

        # Position size
        sl_distance_pct = signal.sl_pct / 100
        position_usd = risk_usd / sl_distance_pct

        # Apply leverage
        position_usd_leveraged = position_usd * signal.leverage

        # Convert to coin quantity
        qty = position_usd_leveraged / signal.entry_price

        # Round to appropriate decimals
        qty = round(qty, 3)

        return qty

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        positions = []

        try:
            response = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )

            if response['retCode'] == 0:
                for pos in response['result']['list']:
                    if float(pos['size']) > 0:
                        positions.append(Position(
                            symbol=pos['symbol'],
                            side=pos['side'],
                            size=float(pos['size']),
                            entry_price=float(pos['avgPrice']),
                            leverage=int(pos['leverage']),
                            unrealized_pnl=float(pos['unrealisedPnl']),
                            take_profit=float(pos['takeProfit']) if pos['takeProfit'] else None,
                            stop_loss=float(pos['stopLoss']) if pos['stopLoss'] else None
                        ))

        except Exception as e:
            print(f"Error getting positions: {e}")

        return positions

    def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol

            response = self.client.cancel_all_orders(**params)
            return response['retCode'] == 0

        except Exception as e:
            print(f"Error cancelling orders: {e}")
            return False
