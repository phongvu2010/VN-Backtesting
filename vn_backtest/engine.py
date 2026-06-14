import pandas as pd
import numpy as np
from typing import Type, Union, List, Dict, Any
from collections import deque
from .strategy import Strategy

class BacktestEngine:
    """
    Backtesting Engine tailored for the Vietnamese Stock Market.
    Supports T+1.5/T+2 settlement cycle, lot size restrictions, transaction costs (taxes & fees),
    and exchange-specific daily price limits (ceiling/floor).
    """
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        initial_cash: float = 100_000_000.0,  # 100M VND default
        buy_fee: float = 0.0015,              # 0.15% brokerage fee
        sell_fee: float = 0.0015,             # 0.15% brokerage fee
        sell_tax: float = 0.001,              # 0.1% selling tax in VN
        settlement_days: int = 2,             # T+2 settlement for shares
        lot_size: int = 100,                  # Lot size of 100 shares in VN
        exchange: str = "hose",               # 'hose' (+/-7%), 'hnx' (+/-10%), 'upcom' (+/-15%)
        execution_at: str = "open",           # Execute at 'open' or 'close' of next bar
        restrict_ceiling_buy: bool = True,    # Cannot buy if price is at Ceiling
        restrict_floor_sell: bool = True,     # Cannot sell if price is at Floor
        slippage: float = 0.0,                # Slippage percentage (e.g. 0.001 = 0.1%)
    ):
        self.data = data.copy()
        self.strategy_class = strategy_class
        self.initial_cash = initial_cash
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.sell_tax = sell_tax
        self.settlement_days = settlement_days
        self.lot_size = lot_size
        self.exchange = exchange.lower()
        self.execution_at = execution_at.lower()
        self.restrict_ceiling_buy = restrict_ceiling_buy
        self.restrict_floor_sell = restrict_floor_sell
        self.slippage = slippage

        # Validate exchange price limit
        if self.exchange == "hose":
            self.price_limit = 0.07
        elif self.exchange == "hnx":
            self.price_limit = 0.10
        elif self.exchange == "upcom":
            self.price_limit = 0.15
        else:
            self.price_limit = 0.0  # No limits

        # Initialize portfolio state
        self.cash = initial_cash
        self.available_cash = initial_cash
        
        # Positions: ticker -> total quantity
        self.positions: Dict[str, int] = {}
        # Sellable shares: ticker -> quantity settled
        self.sellable_shares: Dict[str, int] = {}
        
        # Settlement Queue: list of dicts {'ticker': str, 'quantity': int, 'settle_idx': int}
        self.settlement_queue: List[Dict[str, Any]] = []

        # Order queues
        # Pending orders to execute at next bar: list of dicts
        self.pending_orders: List[Dict[str, Any]] = []

        # History logs
        self.trades_history: List[Dict[str, Any]] = []
        self.order_logs: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []

    def place_buy_order(self, ticker: str, size: Union[float, int, None], time: pd.Timestamp):
        """Queue a buy order for the next bar."""
        self.pending_orders.append({
            'action': 'buy',
            'ticker': ticker,
            'size': size,
            'time_placed': time
        })

    def place_sell_order(self, ticker: str, size: Union[float, int, None], time: pd.Timestamp):
        """Queue a sell order for the next bar."""
        self.pending_orders.append({
            'action': 'sell',
            'ticker': ticker,
            'size': size,
            'time_placed': time
        })

    def _get_execution_price(self, row: pd.Series) -> float:
        """Calculate execution price accounting for slippage."""
        price = row['Open'] if self.execution_at == 'open' else row['Close']
        return price

    def _check_price_limits(self, price: float, prev_close: float) -> tuple[float, float, bool, bool]:
        """
        Calculate ceiling and floor prices and check if execution price hits them.
        Returns: (ceiling, floor, is_ceiling, is_floor)
        """
        if prev_close is None or self.price_limit == 0.0:
            return float('inf'), 0.0, False, False

        # Calculate limits
        ceiling = prev_close * (1 + self.price_limit)
        floor = prev_close * (1 - self.price_limit)
        
        # Round prices (standardizing to 2 decimal places for simple representation)
        ceiling = round(ceiling, 2)
        floor = round(floor, 2)
        
        is_ceiling = price >= ceiling
        is_floor = price <= floor
        
        return ceiling, floor, is_ceiling, is_floor

    def _process_settlements(self, current_idx: int):
        """Move shares from locked to sellable once they reach their settlement index."""
        active_settlements = []
        for item in self.settlement_queue:
            if current_idx >= item['settle_idx']:
                ticker = item['ticker']
                qty = item['quantity']
                self.sellable_shares[ticker] = self.sellable_shares.get(ticker, 0) + qty
            else:
                active_settlements.append(item)
        self.settlement_queue = active_settlements

    def run(self) -> Dict[str, Any]:
        """Run the backtest simulation."""
        # Initialize strategy
        strategy = self.strategy_class(self.data, self)
        strategy.init()

        n_bars = len(self.data)
        
        # Main simulation loop
        for idx in range(n_bars):
            strategy.current_idx = idx
            current_time = self.data.index[idx]
            row = self.data.iloc[idx]
            
            # 1. Process share settlements at start of day
            self._process_settlements(idx)

            # 2. Execute pending orders placed on previous day
            # Reference price for limits is the Close of previous day
            prev_close = self.data.iloc[idx-1]['Close'] if idx > 0 else None
            
            self._execute_orders(row, prev_close, idx, current_time)

            # 3. Calculate portfolio equity at current Close
            equity = self.cash
            for ticker, qty in self.positions.items():
                equity += qty * row['Close']
            
            # Save history
            self.portfolio_history.append({
                'Date': current_time,
                'Cash': self.cash,
                'AvailableCash': self.available_cash,
                'Equity': equity,
                'Close': row['Close']
            })

            # 4. Call strategy's next() to make new trading decisions
            # Strategy has access to current bar's open, high, low, close, volume
            # and can queue new orders for the next bar.
            if idx < n_bars - 1:
                strategy.next()
                
        # Create output DataFrames
        equity_df = pd.DataFrame(self.portfolio_history).set_index('Date')
        trades_df = pd.DataFrame(self.trades_history)
        order_logs_df = pd.DataFrame(self.order_logs)

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'order_logs': order_logs_df,
            'initial_cash': self.initial_cash,
            'final_cash': self.cash,
            'final_equity': equity_df['Equity'].iloc[-1] if not equity_df.empty else self.initial_cash
        }

    def _execute_orders(self, row: pd.Series, prev_close: Union[float, None], current_idx: int, current_time: pd.Timestamp):
        """Execute orders queued on the previous bar."""
        if not self.pending_orders:
            return

        orders_to_process = self.pending_orders.copy()
        self.pending_orders.clear()

        # Simple single stock ticker assumption for the current data
        # If running multi-ticker, row would contain tickers. In our case, we pass the single stock OHLCV.
        # We assume the data corresponds to the requested ticker.
        for order in orders_to_process:
            ticker = order['ticker']
            action = order['action']
            size = order['size']
            time_placed = order['time_placed']

            # Execution Price
            base_price = self._get_execution_price(row)
            
            # Apply Price Limits (Ceiling/Floor)
            ceiling, floor, is_ceiling, is_floor = self._check_price_limits(base_price, prev_close)

            # Check Ceiling/Floor Locks
            if action == 'buy' and is_ceiling and self.restrict_ceiling_buy:
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'BUY_REJECTED',
                    'Reason': f'Price at Ceiling limit ({ceiling})',
                    'Price': base_price,
                    'Quantity': 0
                })
                continue

            if action == 'sell' and is_floor and self.restrict_floor_sell:
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'SELL_REJECTED',
                    'Reason': f'Price at Floor limit ({floor})',
                    'Price': base_price,
                    'Quantity': 0
                })
                continue

            # Limit price execution (cannot buy above ceiling or sell below floor)
            exec_price = base_price
            if exec_price > ceiling:
                exec_price = ceiling
            elif exec_price < floor:
                exec_price = floor

            # Apply Slippage
            if action == 'buy':
                exec_price = exec_price * (1 + self.slippage)
            else:
                exec_price = exec_price * (1 - self.slippage)

            # --- PROCESS BUY ORDER ---
            if action == 'buy':
                # Determine cash allocation
                if size is None:
                    # Max buy
                    cash_to_use = self.available_cash
                elif isinstance(size, float) and 0.0 < size <= 1.0:
                    cash_to_use = self.available_cash * size
                elif isinstance(size, (int, np.integer)) and size >= 1:
                    cash_to_use = size * exec_price * (1 + self.buy_fee)
                else:
                    # Invalid size
                    continue

                if cash_to_use > self.available_cash:
                    # Scale down if we don't have enough cash
                    cash_to_use = self.available_cash

                # Calculate target shares
                target_shares = cash_to_use / (exec_price * (1 + self.buy_fee))
                
                # Round to Lot Size
                if self.lot_size and self.lot_size > 0:
                    qty = int(target_shares // self.lot_size) * self.lot_size
                else:
                    qty = int(target_shares)

                if qty <= 0:
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'BUY_CANCELLED',
                        'Reason': f'Insufficient funds or lot size too small (target: {target_shares:.1f} shares)',
                        'Price': exec_price,
                        'Quantity': 0
                    })
                    continue

                # Execute Buy Trade
                trade_value = qty * exec_price
                fee = trade_value * self.buy_fee
                total_cost = trade_value + fee

                self.cash -= total_cost
                self.available_cash -= total_cost
                
                self.positions[ticker] = self.positions.get(ticker, 0) + qty
                
                # Add to settlement queue
                settle_idx = current_idx + self.settlement_days
                self.settlement_queue.append({
                    'ticker': ticker,
                    'quantity': qty,
                    'settle_idx': settle_idx
                })

                trade_record = {
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'BUY',
                    'Quantity': qty,
                    'Price': exec_price,
                    'Value': trade_value,
                    'Fee': fee,
                    'Tax': 0.0,
                    'TotalValue': total_cost,
                    'TimePlaced': time_placed
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'BUY_FILLED',
                    'Reason': 'Success',
                    'Price': exec_price,
                    'Quantity': qty
                })

            # --- PROCESS SELL ORDER ---
            elif action == 'sell':
                # Get max sellable quantity
                max_sellable = self.sellable_shares.get(ticker, 0)
                
                if max_sellable <= 0:
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'SELL_CANCELLED',
                        'Reason': 'No sellable shares (T+2 lock or no position)',
                        'Price': exec_price,
                        'Quantity': 0
                    })
                    continue

                if size is None:
                    # Sell all
                    qty = max_sellable
                elif isinstance(size, float) and 0.0 < size <= 1.0:
                    target_qty = max_sellable * size
                    # Round down to lot size unless it's the full position
                    if self.lot_size and self.lot_size > 0:
                        qty = int(target_qty // self.lot_size) * self.lot_size
                        if qty == 0 and target_qty > 0 and target_qty == max_sellable:
                            # Sell full odd-lot if it's the entire sellable position
                            qty = max_sellable
                    else:
                        qty = int(target_qty)
                elif isinstance(size, (int, np.integer)) and size >= 1:
                    qty = min(int(size), max_sellable)
                else:
                    continue

                if qty <= 0:
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'SELL_CANCELLED',
                        'Reason': f'Invalid quantity or lot size constraint (sellable: {max_sellable})',
                        'Price': exec_price,
                        'Quantity': 0
                    })
                    continue

                # Execute Sell Trade
                trade_value = qty * exec_price
                fee = trade_value * self.sell_fee
                tax = trade_value * self.sell_tax
                net_proceeds = trade_value - fee - tax

                self.cash += net_proceeds
                self.available_cash += net_proceeds
                
                self.positions[ticker] = self.positions[ticker] - qty
                self.sellable_shares[ticker] = self.sellable_shares[ticker] - qty

                # Clean up empty positions
                if self.positions[ticker] == 0:
                    del self.positions[ticker]
                if self.sellable_shares[ticker] == 0:
                    del self.sellable_shares[ticker]

                trade_record = {
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'SELL',
                    'Quantity': qty,
                    'Price': exec_price,
                    'Value': trade_value,
                    'Fee': fee,
                    'Tax': tax,
                    'TotalValue': net_proceeds,
                    'TimePlaced': time_placed
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'SELL_FILLED',
                    'Reason': 'Success',
                    'Price': exec_price,
                    'Quantity': qty
                })
