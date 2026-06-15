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
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy_class: Type[Strategy],
        initial_cash: float = 100_000_000.0,  # 100M VND default
        buy_fee: float = 0.0015,              # 0.15% brokerage fee
        sell_fee: float = 0.0015,             # 0.15% brokerage fee
        sell_tax: float = 0.001,              # 0.1% selling tax in VN
        settlement_days: int = 2,             # T+2 settlement for shares
        lot_size: int = 100,                  # Lot size of 100 shares in VN
        exchange: Union[str, Dict[str, str]] = "hose", # 'hose', 'hnx', 'upcom'
        execution_at: str = "open",           # Execute at 'open' or 'close' of next bar
        restrict_ceiling_buy: bool = True,    # Cannot buy if price is at Ceiling
        restrict_floor_sell: bool = True,     # Cannot sell if price is at Floor
        slippage: float = 0.0,                # Slippage percentage (e.g. 0.001 = 0.1%)
        dynamic_rules: bool = True,           # Enable dynamic historical rules
        advance_interest_rate: float = 0.12,  # 12% p.a. Cash Advance interest
        auto_close_at_end: bool = True,       # Sell all positions at final bar
    ):
        # Handle multi-ticker data dict
        if isinstance(data, pd.DataFrame):
            ticker = getattr(self, 'ticker', 'FPT')
            self.data = {ticker: data.copy()}
            self.main_ticker = ticker
        elif isinstance(data, dict):
            self.data = {k: v.copy() for k, v in data.items()}
            self.main_ticker = list(self.data.keys())[0] if data else "FPT"
        else:
            raise ValueError("Data must be a pandas DataFrame or a dict of DataFrames")

        self.strategy_class = strategy_class
        self.initial_cash = initial_cash
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.sell_tax = sell_tax
        self.settlement_days = settlement_days
        self.lot_size = lot_size
        self.execution_at = execution_at.lower()
        self.restrict_ceiling_buy = restrict_ceiling_buy
        self.restrict_floor_sell = restrict_floor_sell
        self.slippage = slippage
        self.dynamic_rules = dynamic_rules
        self.advance_interest_rate = advance_interest_rate
        self.auto_close_at_end = auto_close_at_end
        
        # Save default settings to fall back to if dynamic_rules is disabled
        self.default_settlement_days = settlement_days
        self.default_lot_size = lot_size

        # Handle exchange per ticker
        if isinstance(exchange, str):
            self.exchanges = {ticker: exchange.lower() for ticker in self.data}
        elif isinstance(exchange, dict):
            self.exchanges = {ticker: ex.lower() for ticker, ex in exchange.items()}
        else:
            self.exchanges = {ticker: "hose" for ticker in self.data}

        # Align all dates to create unified timeline
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        self.dates = sorted(list(all_dates))

        # Initialize portfolio state
        self.cash = initial_cash
        self.available_cash = initial_cash
        
        # Positions: ticker -> total quantity
        self.positions: Dict[str, int] = {}
        # Sellable shares: ticker -> quantity settled
        self.sellable_shares: Dict[str, int] = {}
        
        # Share Settlement Queue: list of dicts {'ticker': str, 'quantity': int, 'settle_idx': int}
        self.settlement_queue: List[Dict[str, Any]] = []

        # Cash Settlement Queue: list of dicts {'amount': float, 'settle_idx': int, 'borrowed': float}
        self.cash_settlement_queue: List[Dict[str, Any]] = []

        # Order queues
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

    def _round_to_tick(self, price: float, exchange: str, direction: str) -> float:
        """
        Round a price to the nearest tick size according to Vietnam exchange rules.
        Ceiling prices are rounded DOWN to the nearest tick to not exceed the limit.
        Floor prices are rounded UP to the nearest tick.
        """
        price = float(price)
        if exchange == "hose":
            if price < 10000.0:
                tick = 10.0
            elif price < 50000.0:
                tick = 50.0
            else:
                tick = 100.0
        else:
            # HNX and UPCOM use 100 VND tick size for all stocks
            tick = 100.0

        if direction == "down": # Ceiling
            return (price // tick) * tick
        elif direction == "up": # Floor
            return np.ceil(price / tick) * tick
        else:
            return round(price / tick) * tick

    def _check_price_limits(self, price: float, prev_close: float, exchange: str, price_limit: float) -> tuple[float, float, bool, bool]:
        """
        Calculate ceiling and floor prices and check if execution price hits them.
        Returns: (ceiling, floor, is_ceiling, is_floor)
        """
        if prev_close is None or price_limit == 0.0:
            return float('inf'), 0.0, False, False

        # Calculate limits
        raw_ceiling = prev_close * (1 + price_limit)
        raw_floor = prev_close * (1 - price_limit)
        
        ceiling = self._round_to_tick(raw_ceiling, exchange, "down")
        floor = self._round_to_tick(raw_floor, exchange, "up")
        
        is_ceiling = price >= ceiling
        is_floor = price <= floor
        
        return ceiling, floor, is_ceiling, is_floor

    def _get_lot_size(self, ticker: str, current_time: pd.Timestamp) -> int:
        """Get the lot size dynamically based on time and exchange rules."""
        if not self.dynamic_rules:
            return self.default_lot_size
            
        exch = self.exchanges.get(ticker, "hose")
        if exch == "hose":
            if current_time < pd.Timestamp("2021-01-04"):
                return 10
            elif current_time < pd.Timestamp("2022-09-12"):
                return 100
            else:
                return 1
        else:
            return 100

    def _get_price_limit(self, ticker: str) -> float:
        """Get the daily price limit percentage for a stock."""
        exch = self.exchanges.get(ticker, "hose")
        if exch == "hose":
            return 0.07
        elif exch == "hnx":
            return 0.10
        elif exch == "upcom":
            return 0.15
        return 0.0

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

    def _process_cash_settlements(self, current_idx: int):
        """Move cash from pending to available once it reaches settlement index."""
        active_cash_settlements = []
        for item in self.cash_settlement_queue:
            if current_idx >= item['settle_idx']:
                remaining_amount = item['amount'] - item.get('borrowed', 0.0)
                if remaining_amount > 0:
                    self.available_cash += remaining_amount
            else:
                active_cash_settlements.append(item)
        self.cash_settlement_queue = active_cash_settlements

    def _apply_dynamic_rules(self, current_time: pd.Timestamp):
        """Apply VN historical trading rules based on the date."""
        # 1. Settlement cycle: T+3 before 29/08/2022, T+2 from 29/08/2022.
        # However, if execution is at open, shares only arrive at 13:00 on T+2, so they can only be sold at T+3 Open.
        if self.execution_at == 'close':
            if current_time < pd.Timestamp("2022-08-29"):
                self.settlement_days = 3
            else:
                self.settlement_days = 2
        else:  # execution_at == 'open'
            if current_time < pd.Timestamp("2022-08-29"):
                self.settlement_days = 3
            else:
                self.settlement_days = 3

    def run(self) -> Dict[str, Any]:
        """Run the backtest simulation."""
        # Initialize strategy
        strategy = self.strategy_class(self.data, self)
        strategy.init()

        n_bars = len(self.dates)
        
        # Main simulation loop
        for idx in range(n_bars):
            strategy.current_idx = idx
            current_time = self.dates[idx]
            
            # 0. Apply dynamic rules if active
            if self.dynamic_rules:
                self._apply_dynamic_rules(current_time)
            
            # 1. Process share settlements at start of day
            self._process_settlements(idx)
            
            # 2. Process cash settlements at start of day
            self._process_cash_settlements(idx)

            # 3. Execute pending orders placed on previous day
            # Reference prices for limits are the Close of previous day
            prev_closes = {}
            for ticker in self.data:
                ticker_df = self.data[ticker]
                past_df = ticker_df[:current_time]
                if len(past_df) > 1:
                    if current_time in ticker_df.index:
                        prev_closes[ticker] = past_df.iloc[-2]['Close']
                    else:
                        prev_closes[ticker] = past_df.iloc[-1]['Close']
                elif len(past_df) == 1 and current_time not in ticker_df.index:
                    prev_closes[ticker] = past_df.iloc[-1]['Close']
                else:
                    prev_closes[ticker] = None
            
            self._execute_orders(current_time, prev_closes, idx)

            # 4. Calculate portfolio equity at current Close
            equity = self.cash
            for ticker, qty in self.positions.items():
                ticker_df = self.data[ticker]
                if current_time in ticker_df.index:
                    close_price = ticker_df.loc[current_time, 'Close']
                else:
                    past_df = ticker_df[:current_time]
                    close_price = past_df.iloc[-1]['Close'] if not past_df.empty else 0.0
                equity += qty * close_price
            
            # Save history
            self.portfolio_history.append({
                'Date': current_time,
                'Cash': self.cash,
                'AvailableCash': self.available_cash,
                'Equity': equity
            })

            # 5. Call strategy's next() to make new trading decisions
            if idx < n_bars - 1:
                strategy.next()
                
        # Auto-close positions at final bar if active
        if self.auto_close_at_end and any(qty > 0 for qty in self.positions.values()):
            last_date = self.dates[-1]
            active_tickers = [t for t, q in self.positions.items() if q > 0]
            for ticker in active_tickers:
                qty = self.positions[ticker]
                ticker_df = self.data[ticker]
                close_price = ticker_df.loc[last_date, 'Close'] if last_date in ticker_df.index else ticker_df.iloc[-1]['Close']
                
                # Apply transaction fee and tax
                trade_value = qty * close_price
                fee = trade_value * self.sell_fee
                tax = trade_value * self.sell_tax
                net_proceeds = trade_value - fee - tax
                
                # Settle cash immediately since it's the end of backtest
                self.cash += net_proceeds
                self.available_cash += net_proceeds
                
                self.positions[ticker] = 0
                self.sellable_shares[ticker] = 0
                
                trade_record = {
                    'Date': last_date,
                    'Ticker': ticker,
                    'Action': 'SELL',
                    'Quantity': qty,
                    'Price': close_price,
                    'Value': trade_value,
                    'Fee': fee,
                    'Tax': tax,
                    'TotalValue': net_proceeds,
                    'TimePlaced': last_date,
                    'Note': 'Auto-closed at end of backtest'
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': last_date,
                    'Ticker': ticker,
                    'Action': 'SELL_FILLED',
                    'Reason': 'Auto-closed at end of backtest',
                    'Price': close_price,
                    'Quantity': qty
                })
            
            # Update last history record
            self.portfolio_history[-1]['Cash'] = self.cash
            self.portfolio_history[-1]['AvailableCash'] = self.available_cash
            self.portfolio_history[-1]['Equity'] = self.cash

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

    def _execute_orders(self, current_time: pd.Timestamp, prev_closes: Dict[str, float], current_idx: int):
        """Execute orders queued on the previous bar."""
        if not self.pending_orders:
            return

        orders_to_process = self.pending_orders.copy()
        self.pending_orders.clear()

        for order in orders_to_process:
            ticker = order['ticker']
            action = order['action']
            size = order['size']
            time_placed = order['time_placed']

            # Check if ticker traded on current_time
            ticker_df = self.data[ticker]
            if current_time not in ticker_df.index:
                # Keep order pending if ticker didn't trade today
                self.pending_orders.append(order)
                continue

            row = ticker_df.loc[current_time]
            prev_close = prev_closes.get(ticker)
            exch = self.exchanges.get(ticker, "hose")
            lot_size = self._get_lot_size(ticker, current_time)
            price_limit = self._get_price_limit(ticker)

            # Execution Price
            base_price = self._get_execution_price(row)
            
            # Apply Price Limits (Ceiling/Floor)
            ceiling, floor, is_ceiling, is_floor = self._check_price_limits(base_price, prev_close, exch, price_limit)

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
                # Determine maximum buy cash (total cash, including pending)
                max_cash_allowed = self.cash
                
                # Determine cash allocation
                if size is None:
                    cash_to_use = max_cash_allowed
                elif isinstance(size, float) and 0.0 < size <= 1.0:
                    cash_to_use = max_cash_allowed * size
                elif isinstance(size, (int, np.integer)) and size >= 1:
                    cash_to_use = size * exec_price * (1 + self.buy_fee)
                else:
                    continue

                if cash_to_use > max_cash_allowed:
                    cash_to_use = max_cash_allowed

                # Calculate target shares
                target_shares = cash_to_use / (exec_price * (1 + self.buy_fee))
                
                # Round to Lot Size
                if lot_size and lot_size > 0:
                    qty = int(target_shares // lot_size) * lot_size
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

                # Check if we need Cash Advance
                amount_needed = total_cost - self.available_cash
                advance_fee = 0.0
                
                if amount_needed > 0:
                    if total_cost > self.cash:
                        # Scale down buy quantity to fit total cash
                        qty = int(self.cash / (exec_price * (1 + self.buy_fee)))
                        if lot_size and lot_size > 0:
                            qty = int(qty // lot_size) * lot_size
                        
                        if qty <= 0:
                            self.order_logs.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'BUY_CANCELLED',
                                'Reason': f'Insufficient funds including pending cash',
                                'Price': exec_price,
                                'Quantity': 0
                            })
                            continue
                            
                        trade_value = qty * exec_price
                        fee = trade_value * self.buy_fee
                        total_cost = trade_value + fee
                        amount_needed = total_cost - self.available_cash
                    
                    # Borrow from cash_settlement_queue
                    if amount_needed > 0:
                        temp_queue = sorted(self.cash_settlement_queue, key=lambda x: x['settle_idx'])
                        borrowed_so_far = 0.0
                        
                        for item in temp_queue:
                            if borrowed_so_far >= amount_needed:
                                break
                            
                            settle_date = self.dates[min(item['settle_idx'], len(self.dates)-1)]
                            days_diff = (settle_date - current_time).days
                            if days_diff <= 0:
                                days_diff = 1
                                
                            chunk_unborrowed = item['amount'] - item.get('borrowed', 0.0)
                            if chunk_unborrowed <= 0:
                                continue
                                
                            factor = 1.0 + (self.advance_interest_rate / 365.0) * days_diff
                            to_borrow = min(amount_needed - borrowed_so_far, chunk_unborrowed / factor)
                            fee_for_chunk = to_borrow * (self.advance_interest_rate / 365.0) * days_diff
                            
                            item['borrowed'] = item.get('borrowed', 0.0) + to_borrow + fee_for_chunk
                            borrowed_so_far += to_borrow
                            advance_fee += fee_for_chunk

                # Deduct costs from balances
                self.cash -= (total_cost + advance_fee)
                if amount_needed > 0:
                    self.available_cash = 0.0
                else:
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
                    'AdvanceFee': advance_fee,
                    'TimePlaced': time_placed
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': 'BUY_FILLED',
                    'Reason': 'Success' + (f' (Ứng trước, phí: {advance_fee:,.0f}đ)' if advance_fee > 0 else ''),
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
                    qty = max_sellable
                elif isinstance(size, float) and 0.0 < size <= 1.0:
                    target_qty = max_sellable * size
                    if lot_size and lot_size > 0:
                        qty = int(target_qty // lot_size) * lot_size
                        if qty == 0 and target_qty > 0 and target_qty == max_sellable:
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
                
                # Add to cash settlement queue
                settle_idx = current_idx + self.settlement_days
                self.cash_settlement_queue.append({
                    'amount': net_proceeds,
                    'settle_idx': settle_idx,
                    'borrowed': 0.0
                })
                
                self.positions[ticker] = self.positions[ticker] - qty
                self.sellable_shares[ticker] = self.sellable_shares[ticker] - qty

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
