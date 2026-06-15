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
        allow_odd_lot: bool = False,          # Allow odd lot trading (1-99 shares)
        max_volume_ratio: float = None,       # Max trade size as a fraction of daily volume
        adjust_corporate_actions: bool = False, # Set True to simulate corporate actions (splits/dividends)
        margin_ratio: float = 1.0,            # 1.0 = no margin, 0.5 = 50% margin (2x leverage)
        margin_interest_rate: float = 0.13,   # 13% p.a. Margin Loan interest
        margin_maintenance_ratio: float = 0.35, # 35% account equity ratio for margin call liquidation
        ticker: str = None,                   # Optional: Ticker name if data is a single DataFrame
    ):
        # Handle multi-ticker data dict
        if isinstance(data, pd.DataFrame):
            t_name = ticker
            if not t_name:
                t_name = getattr(data, 'name', None)
            if not t_name:
                t_name = getattr(self, 'ticker', 'FPT')
            self.data = {t_name: data.copy()}
            self.main_ticker = t_name
            self.ticker = t_name
        elif isinstance(data, dict):
            self.data = {k: v.copy() for k, v in data.items()}
            self.main_ticker = list(self.data.keys())[0] if data else "FPT"
            self.ticker = ",".join(self.data.keys())
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
        self.allow_odd_lot = allow_odd_lot
        self.max_volume_ratio = max_volume_ratio
        self.adjust_corporate_actions = adjust_corporate_actions
        self.margin_ratio = margin_ratio
        self.margin_interest_rate = margin_interest_rate
        self.margin_maintenance_ratio = margin_maintenance_ratio
        
        # Save default settings to fall back to if dynamic_rules is disabled
        self.default_settlement_days = settlement_days
        self.default_lot_size = lot_size
        self.default_allow_odd_lot = allow_odd_lot

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

        # Fetch corporate actions for all tickers if enabled
        self.corporate_actions = {}
        if self.adjust_corporate_actions:
            from .data import VNStockDataLoader
            loader = VNStockDataLoader()
            for ticker in self.data:
                print(f"-> Đang tải lịch sử cổ tức/chia tách cho {ticker}...")
                self.corporate_actions[ticker] = loader.fetch_corporate_actions(ticker, use_cache=True)

        # Track pending cash dividends: list of dicts {'amount': float, 'payout_date': datetime, 'ticker': str}
        self.pending_dividends: List[Dict[str, Any]] = []

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

    def place_target_percent_order(self, ticker: str, target_percent: float, time: pd.Timestamp):
        """Queue a target percent order for the next bar."""
        self.pending_orders.append({
            'action': 'target_percent',
            'ticker': ticker,
            'target_percent': target_percent,
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
            else:
                return 100
        else:
            return 100

    def _is_odd_lot_allowed(self, ticker: str, current_time: pd.Timestamp) -> bool:
        """Determine if odd-lot trading (1-99 shares) is allowed for a stock today."""
        if not self.dynamic_rules:
            return self.default_allow_odd_lot
        exch = self.exchanges.get(ticker, "hose")
        if exch == "hose":
            return current_time >= pd.Timestamp("2022-08-29")
        return True

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

    def _process_corporate_actions(self, current_time: pd.Timestamp, current_idx: int):
        """Process stock splits, stock dividends, and cash dividends."""
        # 1. Check pending cash dividends payout
        active_pending = []
        for item in self.pending_dividends:
            if current_time.normalize() >= item['payout_date'].normalize():
                amount = item['amount']
                # Apply 5% personal income tax (TNCN) for cash dividends in Vietnam
                dividend_tax_rate = 0.05
                tax = amount * dividend_tax_rate
                net_amount = amount - tax
                
                self.cash += net_amount
                self.available_cash += net_amount
                
                # Log trade record
                self.trades_history.append({
                    'Date': current_time,
                    'Ticker': item['ticker'],
                    'Action': 'DIVIDEND_CASH',
                    'Quantity': 0,
                    'Price': 0.0,
                    'Value': amount,
                    'Fee': 0.0,
                    'Tax': tax,
                    'TotalValue': net_amount,
                    'TimePlaced': current_time,
                    'Note': f"Nhận cổ tức tiền mặt cho {item['ticker']} (Tổng: {amount:,.0f} VND, Thuế 5%: {tax:,.0f} VND, Thực nhận: {net_amount:,.0f} VND)"
                })
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': item['ticker'],
                    'Action': 'DIVIDEND_PAID',
                    'Reason': f"Nhận cổ tức tiền mặt ({amount:,.0f}đ, sau thuế 5%: {net_amount:,.0f}đ)",
                    'Price': 0.0,
                    'Quantity': 0
                })
            else:
                active_pending.append(item)
        self.pending_dividends = active_pending

        # 2. Check for new ex-right events today
        for ticker in self.data:
            actions_df = self.corporate_actions.get(ticker, None)
            if actions_df is None or actions_df.empty:
                continue
                
            # Filter events on this day (normalized to ignore time component)
            events_today = actions_df[actions_df['exright_date'].dt.normalize() == current_time.normalize()]
            for _, event in events_today.iterrows():
                qty = self.positions.get(ticker, 0)
                if qty <= 0:
                    continue
                    
                val_per_share = event.get('value_per_share')
                exercise_ratio = event.get('exercise_ratio')
                event_name = event.get('event_name_vi', '')
                
                # Check for Cash Dividend
                is_cash_div = False
                if pd.notna(val_per_share) and val_per_share > 0:
                    is_cash_div = True
                elif 'tiền mặt' in str(event_name).lower():
                    is_cash_div = True
                    if pd.isna(val_per_share) or val_per_share == 0:
                        # Fallback: estimate 10% par value if exercise_ratio is present
                        if pd.notna(exercise_ratio) and exercise_ratio > 0:
                            val_per_share = exercise_ratio * 10000.0
                        else:
                            val_per_share = 1000.0 # Standard fallback
                
                if is_cash_div:
                    payout_val = val_per_share if pd.notna(val_per_share) else 1000.0
                    dividend_cash = qty * payout_val
                    payout_date = event.get('payout_date')
                    if pd.isna(payout_date) or payout_date is None:
                        payout_date = current_time + pd.Timedelta(days=15)
                    else:
                        payout_date = pd.to_datetime(payout_date)
                        
                    self.pending_dividends.append({
                        'amount': dividend_cash,
                        'payout_date': payout_date,
                        'ticker': ticker
                    })
                    
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'DIVIDEND_ACCRUED',
                        'Reason': f"Chốt quyền nhận cổ tức tiền mặt ({payout_val:,.0f}đ/CP, thanh toán ngày {payout_date.strftime('%d/%m/%Y')})",
                        'Price': payout_val,
                        'Quantity': qty
                    })
                
                # Check for Stock Dividend / Share Issue
                else:
                    ratio = exercise_ratio if pd.notna(exercise_ratio) else 0.0
                    if ratio > 0:
                        new_shares = int(qty * ratio)
                        if new_shares > 0:
                            self.positions[ticker] = self.positions.get(ticker, 0) + new_shares
                            
                            # Determine unlock date
                            unlock_date = event.get('listing_date')
                            if pd.isna(unlock_date) or unlock_date is None:
                                unlock_date = event.get('payout_date')
                            if pd.isna(unlock_date) or unlock_date is None:
                                unlock_date = current_time + pd.Timedelta(days=45)
                            else:
                                unlock_date = pd.to_datetime(unlock_date)
                                
                            # Map unlock_date to trading day index
                            unlock_idx = current_idx + 30
                            for future_idx in range(current_idx, len(self.dates)):
                                if self.dates[future_idx] >= unlock_date:
                                    unlock_idx = future_idx
                                    break
                                    
                            self.settlement_queue.append({
                                'ticker': ticker,
                                'quantity': new_shares,
                                'settle_idx': unlock_idx
                            })
                            
                            self.trades_history.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'DIVIDEND_STOCK',
                                'Quantity': new_shares,
                                'Price': 0.0,
                                'Value': 0.0,
                                'Fee': 0.0,
                                'Tax': 0.0,
                                'TotalValue': 0.0,
                                'TimePlaced': current_time,
                                'Note': f"Nhận cổ tức cổ phiếu tỉ lệ {ratio*100:.1f}% (+{new_shares} CP, mở khóa ngày {self.dates[min(unlock_idx, len(self.dates)-1)].strftime('%d/%m/%Y')})"
                            })
                            self.order_logs.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'DIVIDEND_STOCK_FILLED',
                                'Reason': f"Nhận cổ tức cổ phiếu (+{new_shares} CP)",
                                'Price': 0.0,
                                'Quantity': new_shares
                            })

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
            
            # 2.5 Process corporate actions today if enabled
            if self.adjust_corporate_actions:
                self._process_corporate_actions(current_time, idx)

            # 3. Execute pending orders placed on previous day
            # Reference prices for limits are the Close of previous day (or Weighted Average for UPCoM)
            prev_closes = {}
            for ticker in self.data:
                ticker_df = self.data[ticker]
                exch = self.exchanges.get(ticker, "hose")
                past_df = ticker_df[:current_time]
                
                prev_row = None
                if len(past_df) > 1:
                    if current_time in ticker_df.index:
                        prev_row = past_df.iloc[-2]
                    else:
                        prev_row = past_df.iloc[-1]
                elif len(past_df) == 1 and current_time not in ticker_df.index:
                    prev_row = past_df.iloc[-1]
                    
                if prev_row is not None:
                    if 'Average' in prev_row:
                        prev_closes[ticker] = prev_row['Average']
                    elif exch == 'upcom':
                        # Estimate UPCoM reference price as typical price (Open+High+Low+Close)/4
                        prev_closes[ticker] = (prev_row['Open'] + prev_row['High'] + prev_row['Low'] + prev_row['Close']) / 4.0
                    else:
                        prev_closes[ticker] = prev_row['Close']
                else:
                    prev_closes[ticker] = None
            
            self._execute_orders(current_time, prev_closes, idx)

            # 4. Calculate portfolio equity at current Close
            positions_value = 0.0
            for ticker, qty in self.positions.items():
                ticker_df = self.data[ticker]
                if current_time in ticker_df.index:
                    close_price = ticker_df.loc[current_time, 'Close']
                else:
                    past_df = ticker_df[:current_time]
                    close_price = past_df.iloc[-1]['Close'] if not past_df.empty else 0.0
                positions_value += qty * close_price
            
            equity = self.cash + positions_value
            
            # 4.1 Daily Margin Interest Check
            if self.cash < 0:
                # Calculate actual calendar days elapsed since the previous trading day
                days_diff = 1
                if idx > 0:
                    days_diff = (self.dates[idx] - self.dates[idx-1]).days
                    if days_diff <= 0:
                        days_diff = 1
                interest = abs(self.cash) * (self.margin_interest_rate / 365.0) * days_diff
                self.cash -= interest
                equity -= interest
                
                self.trades_history.append({
                    'Date': current_time,
                    'Ticker': 'MARGIN',
                    'Action': 'MARGIN_INTEREST',
                    'Quantity': 0,
                    'Price': 0.0,
                    'Value': interest,
                    'Fee': interest,
                    'Tax': 0.0,
                    'TotalValue': interest,
                    'TimePlaced': current_time,
                    'Note': f"Lãi vay Margin ({days_diff} ngày): {interest:,.0f} VND (Dư nợ: {abs(self.cash):,.0f} VND)"
                })
                
            # 4.2 Margin Maintenance Ratio Check (Force Sell liquidation)
            if positions_value > 0 and self.margin_ratio < 1.0:
                current_margin_ratio = equity / positions_value
                if current_margin_ratio < self.margin_maintenance_ratio:
                    # Place force sell orders for all positions next day
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': 'PORTFOLIO',
                        'Action': 'FORCE_SELL_TRIGGERED',
                        'Reason': f"Tỷ lệ ký quỹ ({current_margin_ratio*100:.2f}%) < {self.margin_maintenance_ratio*100:.2f}%. Bán giải chấp.",
                        'Price': 0.0,
                        'Quantity': 0
                    })
                    for t in list(self.positions.keys()):
                        self.place_sell_order(t, size=None, time=current_time)
            
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

        # Calculate portfolio equity at start of day for sizing
        start_equity = self.cash
        for ticker, qty in self.positions.items():
            prev_close = prev_closes.get(ticker)
            if prev_close is not None:
                start_equity += qty * prev_close
                
        # 1. Pre-evaluate target percent orders and convert them to buy/sell orders
        evaluated_orders = []
        for order in orders_to_process:
            action = order['action']
            ticker = order['ticker']
            
            # Check if ticker traded on current_time
            ticker_df = self.data[ticker]
            if current_time not in ticker_df.index:
                # Keep order pending if ticker didn't trade today
                self.pending_orders.append(order)
                continue
                
            if action == 'target_percent':
                target_percent = order['target_percent']
                current_qty = self.positions.get(ticker, 0)
                
                if target_percent == 0.0:
                    if current_qty > 0:
                        order['action'] = 'sell'
                        order['size'] = current_qty
                        evaluated_orders.append(order)
                    continue
                    
                row = ticker_df.loc[current_time]
                prev_close = prev_closes.get(ticker)
                exch = self.exchanges.get(ticker, "hose")
                lot_size = self._get_lot_size(ticker, current_time)
                price_limit = self._get_price_limit(ticker)
                
                base_price = self._get_execution_price(row)
                ceiling, floor, is_ceiling, is_floor = self._check_price_limits(base_price, prev_close, exch, price_limit)
                
                # Estimate price to decide direction and quantity
                est_value = current_qty * base_price
                target_value = start_equity * target_percent
                
                if target_value > est_value:
                    # Buy
                    exec_price = base_price * (1 + self.slippage)
                    if exec_price > ceiling: exec_price = ceiling
                    target_shares = target_value / exec_price
                else:
                    # Sell
                    exec_price = base_price * (1 - self.slippage)
                    if exec_price < floor: exec_price = floor
                    target_shares = target_value / exec_price
                    
                effective_lot_size = 1 if self._is_odd_lot_allowed(ticker, current_time) else lot_size
                if effective_lot_size and effective_lot_size > 0:
                    target_shares = int(target_shares // effective_lot_size) * effective_lot_size
                else:
                    target_shares = int(target_shares)
                    
                qty_diff = target_shares - current_qty
                if qty_diff > 0:
                    order['action'] = 'buy'
                    order['size'] = qty_diff
                    evaluated_orders.append(order)
                elif qty_diff < 0:
                    order['action'] = 'sell'
                    order['size'] = abs(qty_diff)
                    evaluated_orders.append(order)
            else:
                evaluated_orders.append(order)
                
        # 2. Sort evaluated orders: SELL first, BUY second to free up cash first
        sorted_orders = sorted(evaluated_orders, key=lambda x: 0 if x['action'] == 'sell' else 1)

        for order in sorted_orders:
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
            
            # Determine dynamic lot size based on odd lot permission
            effective_lot_size = 1 if self._is_odd_lot_allowed(ticker, current_time) else lot_size

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

            # Apply Slippage
            exec_price = base_price
            if action == 'buy':
                exec_price = exec_price * (1 + self.slippage)
            else:
                exec_price = exec_price * (1 - self.slippage)

            # Limit price execution (cannot buy above ceiling or sell below floor)
            if exec_price > ceiling:
                exec_price = ceiling
            elif exec_price < floor:
                exec_price = floor

            # --- PROCESS BUY ORDER ---
            if action == 'buy':
                # Determine cash allocation
                # Calculate Net Equity and max spend for margin trading
                current_positions_value = sum(
                    qty * prev_closes.get(t, 0.0) 
                    for t, qty in self.positions.items() 
                    if prev_closes.get(t) is not None
                )
                net_equity = self.cash + current_positions_value
                max_leverage = 1.0 / self.margin_ratio
                max_spend = max(0.0, net_equity * max_leverage - current_positions_value)

                # Determine cash allocation
                if size is None:
                    cash_to_use = max_spend
                elif isinstance(size, float) and 0.0 < size <= 1.0:
                    cash_to_use = start_equity * size
                elif isinstance(size, (int, np.integer)) and size >= 1:
                    cash_to_use = size * exec_price * (1 + self.buy_fee)
                else:
                    continue

                if cash_to_use > max_spend:
                    cash_to_use = max_spend

                # Calculate target shares
                target_shares = cash_to_use / (exec_price * (1 + self.buy_fee))
                
                # Round to Lot Size
                if effective_lot_size and effective_lot_size > 0:
                    qty = int(target_shares // effective_lot_size) * effective_lot_size
                else:
                    qty = int(target_shares)

                # Apply volume limit constraint if specified
                if self.max_volume_ratio is not None and 'Volume' in row:
                    max_qty = int(row['Volume'] * self.max_volume_ratio)
                    if effective_lot_size and effective_lot_size > 0:
                        max_qty = int(max_qty // effective_lot_size) * effective_lot_size
                    if qty > max_qty:
                        qty = max_qty

                if qty <= 0:
                    reason_msg = f'Insufficient funds or lot size too small (target: {target_shares:.1f} shares)'
                    if self.max_volume_ratio is not None and 'Volume' in row:
                        reason_msg += f' or restricted by volume limit ({int(row["Volume"] * self.max_volume_ratio)} shares)'
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'BUY_CANCELLED',
                        'Reason': reason_msg,
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
                    if total_cost > max_spend:
                        # Scale down buy quantity to fit max_spend
                        max_possible_qty = int(max_spend / (exec_price * (1 + self.buy_fee)))
                        if effective_lot_size and effective_lot_size > 0:
                            max_possible_qty = int(max_possible_qty // effective_lot_size) * effective_lot_size
                        
                        # Adjust qty downwards if advance fee makes cash negative using binary search
                        low_qty = 0
                        high_qty = max_possible_qty
                        best_qty = 0
                        
                        while low_qty <= high_qty:
                            mid_qty = (low_qty + high_qty) // 2
                            if effective_lot_size and effective_lot_size > 0:
                                mid_qty = int(mid_qty // effective_lot_size) * effective_lot_size
                            
                            if mid_qty == 0:
                                break
                                
                            test_trade_value = mid_qty * exec_price
                            test_fee = test_trade_value * self.buy_fee
                            test_total_cost = test_trade_value + test_fee
                            test_amount_needed = test_total_cost - self.available_cash
                            
                            test_advance_fee = 0.0
                            if test_amount_needed > 0:
                                temp_queue = sorted(self.cash_settlement_queue, key=lambda x: x['settle_idx'])
                                borrowed_so_far = 0.0
                                for item in temp_queue:
                                    if borrowed_so_far >= test_amount_needed:
                                        break
                                    settle_date = self.dates[min(item['settle_idx'], len(self.dates)-1)]
                                    days_diff = (settle_date - current_time).days
                                    if days_diff <= 0:
                                        days_diff = 1
                                    chunk_unborrowed = item['amount'] - item.get('borrowed', 0.0)
                                    if chunk_unborrowed <= 0:
                                        continue
                                    factor = 1.0 + (self.advance_interest_rate / 365.0) * days_diff
                                    to_borrow = min(test_amount_needed - borrowed_so_far, chunk_unborrowed / factor)
                                    fee_for_chunk = to_borrow * (self.advance_interest_rate / 365.0) * days_diff
                                    borrowed_so_far += to_borrow
                                    test_advance_fee += fee_for_chunk
                                    
                            if test_total_cost + test_advance_fee <= max_spend:
                                best_qty = mid_qty
                                if effective_lot_size and effective_lot_size > 0:
                                    low_qty = mid_qty + effective_lot_size
                                else:
                                    low_qty = mid_qty + 1
                            else:
                                if effective_lot_size and effective_lot_size > 0:
                                    high_qty = mid_qty - effective_lot_size
                                else:
                                    high_qty = mid_qty - 1
                                    
                        qty = best_qty
                        
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
                    if effective_lot_size and effective_lot_size > 0:
                        qty = int(target_qty // effective_lot_size) * effective_lot_size
                        if qty == 0 and target_qty > 0 and target_qty == max_sellable:
                            qty = max_sellable
                    else:
                        qty = int(target_qty)
                elif isinstance(size, (int, np.integer)) and size >= 1:
                    qty = min(int(size), max_sellable)
                else:
                    continue

                # Apply volume limit constraint if specified
                if self.max_volume_ratio is not None and 'Volume' in row:
                    max_qty = int(row['Volume'] * self.max_volume_ratio)
                    if effective_lot_size and effective_lot_size > 0:
                        max_qty = int(max_qty // effective_lot_size) * effective_lot_size
                    if qty > max_qty:
                        qty = max_qty

                if qty <= 0:
                    reason_msg = f'Invalid quantity or lot size constraint (sellable: {max_sellable})'
                    if self.max_volume_ratio is not None and 'Volume' in row:
                        reason_msg += f' or restricted by volume limit ({int(row["Volume"] * self.max_volume_ratio)} shares)'
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'SELL_CANCELLED',
                        'Reason': reason_msg,
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
