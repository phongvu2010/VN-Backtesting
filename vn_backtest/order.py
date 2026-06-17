"""
Order Management System for VN-Backtesting.

Provides full Order lifecycle management with support for:
- Market, Limit, Stop, and Stop-Limit order types
- Order lifecycle: PENDING → PARTIALLY_FILLED → FILLED / CANCELLED / EXPIRED
- OCO (One-Cancels-Other) groups for Take-Profit + Stop-Loss combos
- Order expiry (Good-Till-Cancelled or N-bar expiry)

Inspired by Backtrader's order management architecture.
"""

from enum import Enum
from typing import Optional, List, Dict
import pandas as pd


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = 'PENDING'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'
    FILLED = 'FILLED'
    CANCELLED = 'CANCELLED'
    EXPIRED = 'EXPIRED'


class OrderType(Enum):
    """Order type classification."""
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    STOP_LIMIT = 'STOP_LIMIT'


class Order:
    """
    Represents a trading order with full lifecycle management.

    Lifecycle Flow:
        PENDING → PARTIALLY_FILLED → FILLED
                ↘ CANCELLED
                ↘ EXPIRED

    Attributes:
        order_id (int): Unique order identifier (auto-assigned by engine).
        ticker (str): The stock ticker symbol.
        action (str): 'buy', 'sell', or 'target_percent'.
        order_type (OrderType): MARKET, LIMIT, STOP, or STOP_LIMIT.
        quantity (int): Target number of shares (set after sizing).
        filled_quantity (int): Number of shares already filled.
        limit_price (float|None): Limit price for LIMIT/STOP_LIMIT orders.
        stop_price (float|None): Stop trigger price for STOP/STOP_LIMIT orders.
        status (OrderStatus): Current lifecycle status.
        expiry_bars (int|None): Number of bars before expiry (None = GTC).
        oco_group_id (int|None): ID of the OCO group this order belongs to.
        size (float|None): Abstract size before sizing (fraction or int).
        target_percent (float|None): Target allocation for target_percent orders.
    """

    __slots__ = (
        'order_id', 'ticker', 'action', 'order_type', 'quantity',
        'limit_price', 'stop_price', 'status', 'filled_quantity',
        'avg_fill_price', 'created_at', 'updated_at', 'expiry_bars',
        'bars_alive', 'oco_group_id', 'size', 'target_percent',
        'time_placed', 'is_sized', 'applied_slippage', '_stop_triggered',
    )

    def __init__(
        self,
        order_id: int,
        ticker: str,
        action: str,
        order_type: OrderType = OrderType.MARKET,
        quantity: int = 0,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        created_at: Optional[pd.Timestamp] = None,
        expiry_bars: Optional[int] = None,
        oco_group_id: Optional[int] = None,
        size=None,
        target_percent: Optional[float] = None,
        time_placed: Optional[pd.Timestamp] = None,
    ):
        self.order_id = order_id
        self.ticker = ticker
        self.action = action
        self.order_type = order_type
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.avg_fill_price = 0.0
        self.created_at = created_at
        self.updated_at = created_at
        self.expiry_bars = expiry_bars
        self.bars_alive = 0
        self.oco_group_id = oco_group_id
        self.size = size
        self.target_percent = target_percent
        self.time_placed = time_placed or created_at
        self.is_sized = (quantity > 0 and size is None and target_percent is None)
        self.applied_slippage = 0.0
        self._stop_triggered = False

    @property
    def remaining_quantity(self) -> int:
        """Number of shares still to be filled."""
        return max(0, self.quantity - self.filled_quantity)

    @property
    def is_active(self) -> bool:
        """True if order can still receive fills."""
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)

    def fill(self, qty: int, price: float) -> None:
        """
        Record a partial or full fill.

        Args:
            qty: Number of shares filled in this execution.
            price: Execution price for this fill.
        """
        if qty <= 0:
            return
        # Update weighted average fill price
        total_value = self.avg_fill_price * self.filled_quantity + price * qty
        self.filled_quantity += qty
        self.avg_fill_price = total_value / self.filled_quantity

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_quantity = self.quantity
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel this order if it is still active."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED

    def expire(self) -> None:
        """Mark this order as expired if it is still active."""
        if self.is_active:
            self.status = OrderStatus.EXPIRED

    def check_stop_trigger(self, high: float, low: float) -> bool:
        """
        Check if a Stop order's trigger condition is met by the current bar's range.

        For SELL stops: triggers when price drops to or below stop_price (Low <= stop_price).
        For BUY stops: triggers when price rises to or above stop_price (High >= stop_price).

        Returns:
            True if the stop has been triggered (now or previously), False otherwise.
        """
        if self.order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
            return False
        if self.stop_price is None:
            return False
        if self._stop_triggered:
            return True

        triggered = False
        if self.action == 'sell':
            triggered = low <= self.stop_price
        elif self.action == 'buy':
            triggered = high >= self.stop_price

        if triggered:
            self._stop_triggered = True
        return triggered

    def tick(self) -> bool:
        """
        Increment the bar counter. Call at the start of each execution attempt.

        Returns:
            True if the order expired (bars_alive > expiry_bars), False otherwise.
        """
        self.bars_alive += 1
        if self.expiry_bars is not None and self.bars_alive > self.expiry_bars:
            self.expire()
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, {self.action.upper()} {self.ticker}, "
            f"type={self.order_type.value}, qty={self.quantity}, "
            f"filled={self.filled_quantity}, status={self.status.value})"
        )


class OCOGroup:
    """
    One-Cancels-Other order group.

    Groups two or more orders together. When any order in the group is filled,
    all other orders in the group are automatically cancelled.

    Typical use case: Pair a Take-Profit (Limit Sell) with a Stop-Loss (Stop Sell).
    Whichever triggers first cancels the other.
    """

    __slots__ = ('group_id', 'order_ids', 'is_active')

    def __init__(self, group_id: int, order_ids: List[int]):
        self.group_id = group_id
        self.order_ids = list(order_ids)
        self.is_active = True

    def on_fill(self, filled_order_id: int, all_orders: Dict[int, 'Order']) -> List[int]:
        """
        Handle a fill event for one order in the group.
        Cancels all other orders in the group.

        Args:
            filled_order_id: The ID of the order that was filled.
            all_orders: Reference to the engine's all_orders dict.

        Returns:
            List of order IDs that were cancelled.
        """
        if not self.is_active:
            return []

        cancelled = []
        for oid in self.order_ids:
            if oid != filled_order_id and oid in all_orders:
                order = all_orders[oid]
                if order.is_active:
                    order.cancel()
                    cancelled.append(oid)

        self.is_active = False
        return cancelled

    def __repr__(self) -> str:
        return f"OCOGroup(id={self.group_id}, orders={self.order_ids}, active={self.is_active})"
