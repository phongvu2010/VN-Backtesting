import pandas as pd
import numpy as np
from typing import Type, Union, List, Dict, Any, Optional
from collections import deque
from .strategy import Strategy
from .order import Order, OrderType, OrderStatus, OCOGroup

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
        corporate_actions: Dict[str, pd.DataFrame] = None,
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
        force_adjusted: bool = None,          # Force adjusted status (True: already adjusted, False: raw)
        margin_ratio: float = 1.0,            # 1.0 = no margin, 0.5 = 50% margin (2x leverage)
        margin_interest_rate: float = 0.13,   # 13% p.a. Margin Loan interest
        margin_maintenance_ratio: float = 0.35, # 35% account equity ratio for margin call liquidation
        ticker: str = None,                   # Optional: Ticker name if data is a single DataFrame
        strategy_params: Dict[str, Any] = None, # Dict of parameters for strategy init
        market_impact_coef: float = 0.0,      # Market impact coefficient for dynamic slippage
        rights_listing_delay: int = 90,       # Calendar days delay for rights/stock dividend listing
        dividend_tax_rate: float = 0.05,      # 5% dividend tax rate in VN
        partial_fill_mode: str = 'defer',     # 'defer' = pending orders carry over, 'cancel' = unfilled part cancelled
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

        self.corporate_actions = corporate_actions or {}
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
        self.market_impact_coef = market_impact_coef
        self.dynamic_rules = dynamic_rules
        self.advance_interest_rate = advance_interest_rate
        self.auto_close_at_end = auto_close_at_end
        self.allow_odd_lot = allow_odd_lot
        self.max_volume_ratio = max_volume_ratio
        self.adjust_corporate_actions = adjust_corporate_actions
        self.force_adjusted = force_adjusted
        self.margin_ratio = margin_ratio
        self.margin_interest_rate = margin_interest_rate
        self.margin_maintenance_ratio = margin_maintenance_ratio
        self.strategy_params = strategy_params or {}
        self.rights_listing_delay = rights_listing_delay
        self.dividend_tax_rate = dividend_tax_rate
        self.dividend_shares = {}
        self.partial_fill_mode = partial_fill_mode
        
        # Order Management System
        self.order_id_counter: int = 0
        self.oco_group_counter: int = 0
        self.oco_groups: Dict[int, OCOGroup] = {}
        self.all_orders: Dict[int, Order] = {}  # Track all orders ever created
        
        # Risk management tracking
        self.position_entry_price = {}
        self.position_highest_price = {}
        
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
        self.corporate_actions = corporate_actions or {}

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

        # Order queues (Order objects)
        self.pending_orders: List[Order] = []

        # History logs
        self.trades_history: List[Dict[str, Any]] = []
        self.order_logs: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []

        # Detect if data is already adjusted
        self.dividends_already_factored = False
        if self.adjust_corporate_actions:
            if self.force_adjusted is not None:
                self.dividends_already_factored = self.force_adjusted
            else:
                self.dividends_already_factored = self._detect_if_adjusted()
            
            if self.dividends_already_factored:
                print("==============================================================")
                print("Cáº¢NH BÃO: PhÃ¡t hiá»‡n dá»¯ liá»‡u giÃ¡ Ä‘áº§u vÃ o Ä‘Ã£ Ä‘Æ°á»£c ÄIá»€U CHá»ˆNH (Adjusted).")
                print("Tá»± Ä‘á»™ng vÃ´ hiá»‡u hÃ³a viá»‡c cá»™ng dá»“n cá»• tá»©c/chia tÃ¡ch Ä‘á»ƒ trÃ¡nh lá»—i Double-Adjustment.")
                print("==============================================================")

        # Calculate adjusted price columns
        self._calculate_adjusted_prices()

        # Identify first listing dates before reindexing
        self.raw_listing_dates = {}
        for ticker, df in self.data.items():
            valid_df = df.dropna(subset=['Close'])
            if not valid_df.empty:
                self.raw_listing_dates[ticker] = valid_df.index[0]

        # Reindex and fill data to prevent multi-ticker timeline alignment issues
        self._reindex_and_fill_data()

    def _next_order_id(self) -> int:
        """Generate a unique order ID."""
        self.order_id_counter += 1
        return self.order_id_counter

    def place_buy_order(self, ticker: str, size: Union[float, int, None], time: pd.Timestamp,
                        limit_price: float = None, stop_price: float = None,
                        expiry_bars: int = None, oco_group_id: int = None) -> int:
        """Queue a buy order for the next bar. Returns the order ID."""
        if stop_price is not None:
            otype = OrderType.STOP_LIMIT if limit_price is not None else OrderType.STOP
        elif limit_price is not None:
            otype = OrderType.LIMIT
        else:
            otype = OrderType.MARKET

        order = Order(
            order_id=self._next_order_id(),
            ticker=ticker, action='buy', order_type=otype,
            size=size, limit_price=limit_price, stop_price=stop_price,
            created_at=time, time_placed=time,
            expiry_bars=expiry_bars, oco_group_id=oco_group_id,
        )
        self.pending_orders.append(order)
        self.all_orders[order.order_id] = order
        return order.order_id

    def place_sell_order(self, ticker: str, size: Union[float, int, None], time: pd.Timestamp,
                         limit_price: float = None, stop_price: float = None,
                         expiry_bars: int = None, oco_group_id: int = None) -> int:
        """Queue a sell order for the next bar. Returns the order ID."""
        if stop_price is not None:
            otype = OrderType.STOP_LIMIT if limit_price is not None else OrderType.STOP
        elif limit_price is not None:
            otype = OrderType.LIMIT
        else:
            otype = OrderType.MARKET

        order = Order(
            order_id=self._next_order_id(),
            ticker=ticker, action='sell', order_type=otype,
            size=size, limit_price=limit_price, stop_price=stop_price,
            created_at=time, time_placed=time,
            expiry_bars=expiry_bars, oco_group_id=oco_group_id,
        )
        self.pending_orders.append(order)
        self.all_orders[order.order_id] = order
        return order.order_id

    def place_target_percent_order(self, ticker: str, target_percent: float, time: pd.Timestamp) -> int:
        """Queue a target percent order for the next bar. Returns the order ID."""
        order = Order(
            order_id=self._next_order_id(),
            ticker=ticker, action='target_percent', order_type=OrderType.MARKET,
            target_percent=target_percent,
            created_at=time, time_placed=time,
        )
        self.pending_orders.append(order)
        self.all_orders[order.order_id] = order
        return order.order_id

    def place_stop_order(self, ticker: str, size: Union[float, int, None], stop_price: float,
                         time: pd.Timestamp, limit_price: float = None,
                         expiry_bars: int = None) -> int:
        """
        Place a Stop or Stop-Limit order.
        
        - Stop Order (limit_price=None): Triggers as Market order when price hits stop_price.
        - Stop-Limit Order (limit_price set): Triggers as Limit order when price hits stop_price.

        Args:
            ticker: Stock ticker.
            size: Order size (shares or fraction).
            stop_price: The trigger price level.
            time: Timestamp of the order placement.
            limit_price: Optional limit price after stop triggers.
            expiry_bars: Number of bars before order expires (None = GTC).

        Returns:
            int: The order ID.
        """
        action = 'sell'  # Most stops are sell stops; buy stops are rare
        otype = OrderType.STOP_LIMIT if limit_price is not None else OrderType.STOP
        order = Order(
            order_id=self._next_order_id(),
            ticker=ticker, action=action, order_type=otype,
            size=size, stop_price=stop_price, limit_price=limit_price,
            created_at=time, time_placed=time, expiry_bars=expiry_bars,
        )
        self.pending_orders.append(order)
        self.all_orders[order.order_id] = order
        return order.order_id

    def place_oco_orders(self, ticker: str, size: Union[float, int, None],
                         take_profit_price: float, stop_loss_price: float,
                         time: pd.Timestamp, expiry_bars: int = None) -> tuple:
        """
        Place an OCO (One-Cancels-Other) pair: Take-Profit + Stop-Loss.
        When one order fills, the other is automatically cancelled.

        Args:
            ticker: Stock ticker.
            size: Number of shares to sell.
            take_profit_price: Limit sell price for taking profit.
            stop_loss_price: Stop trigger price for cutting losses.
            time: Timestamp of the order placement.
            expiry_bars: Number of bars before both orders expire (None = GTC).

        Returns:
            tuple: (take_profit_order_id, stop_loss_order_id)
        """
        self.oco_group_counter += 1
        gid = self.oco_group_counter

        tp_id = self.place_sell_order(
            ticker, size, time, limit_price=take_profit_price,
            expiry_bars=expiry_bars, oco_group_id=gid
        )
        sl_id = self.place_sell_order(
            ticker, size, time, stop_price=stop_loss_price,
            expiry_bars=expiry_bars, oco_group_id=gid
        )

        self.oco_groups[gid] = OCOGroup(gid, [tp_id, sl_id])
        return (tp_id, sl_id)

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a pending order by its ID.

        Returns:
            True if the order was successfully cancelled, False if not found or already inactive.
        """
        if order_id in self.all_orders:
            order = self.all_orders[order_id]
            if order.is_active:
                order.cancel()
                self.pending_orders = [o for o in self.pending_orders if o.order_id != order_id]
                return True
        return False

    def cancel_all_orders(self, ticker: str = None) -> int:
        """
        Cancel all pending orders, optionally filtered by ticker.

        Returns:
            Number of orders cancelled.
        """
        count = 0
        remaining = []
        for order in self.pending_orders:
            if ticker is None or order.ticker == ticker:
                if order.is_active:
                    order.cancel()
                    count += 1
            else:
                remaining.append(order)
        if ticker is not None:
            self.pending_orders = remaining
        else:
            self.pending_orders.clear()
        return count

    def get_active_orders(self, ticker: str = None) -> List[Order]:
        """
        Get all currently active (pending/partially filled) orders.

        Args:
            ticker: Optional filter by ticker.

        Returns:
            List of active Order objects.
        """
        orders = [o for o in self.pending_orders if o.is_active]
        if ticker is not None:
            orders = [o for o in orders if o.ticker == ticker]
        return orders

    def _get_execution_price(self, row: pd.Series) -> float:
        """Calculate base execution price according to the execution model."""
        exec_mode = self.execution_at.lower()
        if exec_mode == 'open':
            return float(row['Open'])
        elif exec_mode == 'close':
            return float(row['Close'])
        elif exec_mode in ['average', 'vwap']:
            if 'Average' in row and pd.notna(row['Average']) and row['Average'] > 0:
                return float(row['Average'])
            # fallback to OHLC average
            return float((row['Open'] + row['High'] + row['Low'] + row['Close']) / 4.0)
        elif exec_mode == 'hl2':
            return float((row['High'] + row['Low']) / 2.0)
        elif exec_mode == 'typical':
            return float((row['High'] + row['Low'] + row['Close']) / 3.0)
        else:
            # default fallback
            return float(row['Open'] if exec_mode == 'open' else row['Close'])

    def _get_tick_size(self, price: float, exchange: str, current_time: pd.Timestamp = None) -> float:
        """Get the tick size for a given price according to exchange rules."""
        price = float(price)
        if exchange == "hose":
            if self.dynamic_rules and current_time is not None and current_time < pd.Timestamp("2016-09-12"):
                # HOSE rules before 12/09/2016
                if price < 50000.0:
                    return 100.0
                elif price < 100000.0:
                    return 500.0
                else:
                    return 1000.0
            else:
                # HOSE rules from 12/09/2016 onwards
                if price < 10000.0:
                    return 10.0
                elif price < 50000.0:
                    return 50.0
                else:
                    return 100.0
        else:
            # HNX and UPCOM use 100 VND tick size for all stocks
            return 100.0

    def _round_to_tick(self, price: float, exchange: str, direction: str, current_time: pd.Timestamp = None) -> float:
        """
        Round a price to the nearest tick size according to Vietnam exchange rules.
        Ceiling prices are rounded DOWN to the nearest tick to not exceed the limit.
        Floor prices are rounded UP to the nearest tick.
        HOSE uses historical tick sizes if dynamic rules are enabled.
        """
        price = float(price)
        tick = self._get_tick_size(price, exchange, current_time)

        if direction == "down": # Ceiling
            return (price // tick) * tick
        elif direction == "up": # Floor
            return np.ceil(price / tick) * tick
        else:
            return round(price / tick) * tick

    def _check_price_limits(self, price: float, prev_close: float, exchange: str, price_limit: float, current_time: pd.Timestamp = None) -> tuple[float, float, bool, bool]:
        """
        Calculate ceiling and floor prices and check if execution price hits them.
        Returns: (ceiling, floor, is_ceiling, is_floor)
        """
        if prev_close is None or price_limit == 0.0:
            return float('inf'), 0.0, False, False

        # Calculate limits
        raw_ceiling = prev_close * (1 + price_limit)
        raw_floor = prev_close * (1 - price_limit)
        
        ceiling = self._round_to_tick(raw_ceiling, exchange, "down", current_time)
        floor = self._round_to_tick(raw_floor, exchange, "up", current_time)
        
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
            return current_time >= pd.Timestamp("2022-09-12")
        return True

    def _get_price_limit(self, ticker: str, current_time: pd.Timestamp) -> float:
        """Get the daily price limit percentage for a stock based on date and listing status."""
        exch = self.exchanges.get(ticker, "hose")
        
        # Check if today is the listing day
        is_listing_day = False
        if hasattr(self, 'raw_listing_dates') and ticker in self.raw_listing_dates:
            if current_time.normalize() == self.raw_listing_dates[ticker].normalize():
                is_listing_day = True
                
        if is_listing_day:
            if exch == "hose":
                return 0.20
            elif exch == "hnx":
                return 0.30
            elif exch == "upcom":
                return 0.40
            return 0.0

        # Normal trading day - historical limits
        if exch == "hose":
            if current_time < pd.Timestamp("2000-08-24"):
                return 0.02
            elif current_time < pd.Timestamp("2001-06-13"):
                return 0.05
            elif current_time < pd.Timestamp("2002-08-01"):
                return 0.02
            elif current_time < pd.Timestamp("2003-01-02"):
                return 0.03
            elif current_time < pd.Timestamp("2008-03-27"):
                return 0.05
            elif current_time < pd.Timestamp("2008-04-07"):
                return 0.01
            elif current_time < pd.Timestamp("2008-06-19"):
                return 0.02
            elif current_time < pd.Timestamp("2008-08-18"):
                return 0.03
            elif current_time < pd.Timestamp("2013-01-15"):
                return 0.05
            else:
                return 0.07
        elif exch == "hnx":
            if current_time < pd.Timestamp("2008-03-27"):
                return 0.10
            elif current_time < pd.Timestamp("2008-04-07"):
                return 0.02
            elif current_time < pd.Timestamp("2008-06-19"):
                return 0.03
            elif current_time < pd.Timestamp("2008-08-18"):
                return 0.05
            elif current_time < pd.Timestamp("2013-01-15"):
                return 0.07
            else:
                return 0.10
        elif exch == "upcom":
            if current_time < pd.Timestamp("2015-07-01"):
                return 0.10
            else:
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
        if self.dividends_already_factored:
            return
        # 1. Check pending cash dividends payout
        active_pending = []
        for item in self.pending_dividends:
            payout_dt = item['payout_date']
            if payout_dt.tz is not None:
                payout_dt = payout_dt.tz_localize(None)
            if current_time.normalize() >= payout_dt.normalize():
                amount = item['amount']
                tax = amount * self.dividend_tax_rate
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
                    'Note': f"Nháº­n cá»• tá»©c tiá»n máº·t cho {item['ticker']} (Tá»•ng: {amount:,.0f} VND, Thuáº¿ {self.dividend_tax_rate*100:.1f}%: {tax:,.0f} VND, Thá»±c nháº­n: {net_amount:,.0f} VND)"
                })
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': item['ticker'],
                    'Action': 'DIVIDEND_PAID',
                    'Reason': f"Nháº­n cá»• tá»©c tiá»n máº·t ({amount:,.0f}Ä‘, sau thuáº¿ {self.dividend_tax_rate*100:.1f}%: {net_amount:,.0f}Ä‘)",
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
            if actions_df['exright_date'].dt.tz is not None:
                actions_df['exright_date'] = actions_df['exright_date'].dt.tz_localize(None)
            events_today = actions_df[actions_df['exright_date'].dt.normalize() == current_time.normalize()]
            for _, event in events_today.iterrows():
                qty = self.positions.get(ticker, 0)
                if qty <= 0:
                    continue
                    
                val_per_share = event.get('value_per_share')
                exercise_ratio = event.get('exercise_ratio')
                event_name = event.get('event_name_vi', '')
                event_title = event.get('event_title_vi', '') if 'event_title_vi' in event.index else ''
                
                # Check for Cash Dividend
                is_cash_div = False
                if pd.notna(val_per_share) and val_per_share > 0:
                    is_cash_div = True
                elif 'tiá»n máº·t' in str(event_name).lower() or 'tiá»n máº·t' in str(event_title).lower():
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
                        if payout_date.tz is not None:
                            payout_date = payout_date.tz_localize(None)
                        
                    self.pending_dividends.append({
                        'amount': dividend_cash,
                        'payout_date': payout_date,
                        'ticker': ticker
                    })
                    
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'DIVIDEND_ACCRUED',
                        'Reason': f"Chá»‘t quyá»n nháº­n cá»• tá»©c tiá»n máº·t ({payout_val:,.0f}Ä‘/CP, thanh toÃ¡n ngÃ y {payout_date.strftime('%d/%m/%Y')})",
                        'Price': payout_val,
                        'Quantity': qty
                    })
                    
                    # Adjust risk management prices for cash dividend ex-rights price drop
                    close_prev = self.data[ticker].iloc[current_idx - 1]['Close'] if current_idx > 0 else payout_val
                    if close_prev > payout_val:
                        factor = (close_prev - payout_val) / close_prev
                        if ticker in self.position_entry_price:
                            self.position_entry_price[ticker] *= factor
                        if ticker in self.position_highest_price:
                            self.position_highest_price[ticker] *= factor
                
                # Check for Rights Offering (Quyá»n mua phÃ¡t hÃ nh thÃªm)
                elif 'quyá»n mua' in str(event_title).lower() or 'quyá»n mua' in str(event_name).lower():
                    ratio = exercise_ratio if pd.notna(exercise_ratio) else 0.0
                    if ratio > 0:
                        # Adjust risk management prices for dilution from rights issue ex-rights price drop
                        close_prev = self.data[ticker].iloc[current_idx - 1]['Close'] if current_idx > 0 else 10000.0
                        if close_prev > 10000.0:
                            factor = (close_prev + ratio * 10000.0) / (close_prev * (1.0 + ratio))
                            if ticker in self.position_entry_price:
                                self.position_entry_price[ticker] *= factor
                            if ticker in self.position_highest_price:
                                self.position_highest_price[ticker] *= factor

                        new_shares = int(qty * ratio)
                        if new_shares > 0:
                            # Check if market price is above subscription price (usually 10,000 VND)
                            subscription_price = 10000.0
                            ticker_df = self.data[ticker]
                            current_price = ticker_df.loc[current_time, 'Close'] if current_time in ticker_df.index else subscription_price
                            
                            if current_price > subscription_price:
                                cash_needed = new_shares * subscription_price
                                
                                # Check cash restriction
                                can_exercise = True
                                if self.margin_ratio >= 1.0 and cash_needed > self.cash:
                                    # Limit new shares to cash capacity if margin not allowed
                                    max_buyable = int(self.cash // subscription_price)
                                    if max_buyable > 0:
                                        new_shares = min(new_shares, max_buyable)
                                        cash_needed = new_shares * subscription_price
                                    else:
                                        can_exercise = False
                                        
                                if can_exercise:
                                                                     # Determine unlock date (usually listing date or self.rights_listing_delay days)
                                    unlock_date = event.get('listing_date')
                                    if pd.isna(unlock_date) or unlock_date is None:
                                        unlock_date = event.get('payout_date')
                                    if pd.isna(unlock_date) or unlock_date is None:
                                        unlock_date = current_time + pd.Timedelta(days=self.rights_listing_delay)
                                    else:
                                        unlock_date = pd.to_datetime(unlock_date)
                                        if unlock_date.tz is not None:
                                            unlock_date = unlock_date.tz_localize(None)
                                         
                                    unlock_idx = current_idx + int(self.rights_listing_delay * 5 / 7)
                                    for future_idx in range(current_idx, len(self.dates)):
                                        target_dt = self.dates[future_idx]
                                        if target_dt.tz is not None:
                                            target_dt = target_dt.tz_localize(None)
                                        if target_dt >= unlock_date:
                                            unlock_idx = future_idx
                                            break
                                            
                                    self.positions[ticker] = self.positions.get(ticker, 0) + new_shares
                                    
                                    self.settlement_queue.append({
                                        'ticker': ticker,
                                        'quantity': new_shares,
                                        'settle_idx': unlock_idx
                                    })
                                    
                                    self.trades_history.append({
                                        'Date': current_time,
                                        'Ticker': ticker,
                                        'Action': 'RIGHTS_EXERCISED',
                                        'Quantity': new_shares,
                                        'Price': subscription_price,
                                        'Value': cash_needed,
                                        'Fee': 0.0,
                                        'Tax': 0.0,
                                        'TotalValue': cash_needed,
                                        'TimePlaced': current_time,
                                        'Note': f"Thá»±c hiá»‡n quyá»n mua tá»‰ lá»‡ {ratio*100:.1f}% giÃ¡ {subscription_price:,.0f}Ä‘ (-{cash_needed:,.0f} VND, +{new_shares} CP, má»Ÿ khÃ³a ngÃ y {self.dates[min(unlock_idx, len(self.dates)-1)].strftime('%d/%m/%Y')})"
                                    })
                                    self.order_logs.append({
                                        'Date': current_time,
                                        'Ticker': ticker,
                                        'Action': 'RIGHTS_EXERCISED',
                                        'Reason': f"Thá»±c hiá»‡n quyá»n mua (+{new_shares} CP @ {subscription_price:,.0f}Ä‘)",
                                        'Price': subscription_price,
                                        'Quantity': new_shares
                                    })
                                else:
                                    self.order_logs.append({
                                        'Date': current_time,
                                        'Ticker': ticker,
                                        'Action': 'RIGHTS_LAPSED',
                                        'Reason': 'Insufficient cash to exercise rights',
                                        'Price': subscription_price,
                                        'Quantity': 0
                                    })
                            else:
                                self.order_logs.append({
                                    'Date': current_time,
                                    'Ticker': ticker,
                                    'Action': 'RIGHTS_LAPSED',
                                    'Reason': f"Market price ({current_price:,.0f}Ä‘) <= rights price ({subscription_price:,.0f}Ä‘)",
                                    'Price': subscription_price,
                                    'Quantity': 0
                                })
                # Check for Stock Dividend / Share Issue (free)
                else:
                    ratio = exercise_ratio if pd.notna(exercise_ratio) else 0.0
                    if ratio > 0:
                        # Adjust risk management prices for stock dividend ex-rights price drop
                        factor = 1.0 / (1.0 + ratio)
                        if ticker in self.position_entry_price:
                            self.position_entry_price[ticker] *= factor
                        if ticker in self.position_highest_price:
                            self.position_highest_price[ticker] *= factor
 
                        new_shares = int(qty * ratio)
                        if new_shares > 0:
                            self.positions[ticker] = self.positions.get(ticker, 0) + new_shares
                            self.dividend_shares[ticker] = self.dividend_shares.get(ticker, 0) + new_shares
                            
                            # Determine unlock date
                            unlock_date = event.get('listing_date')
                            if pd.isna(unlock_date) or unlock_date is None:
                                unlock_date = event.get('payout_date')
                            if pd.isna(unlock_date) or unlock_date is None:
                                unlock_date = current_time + pd.Timedelta(days=self.rights_listing_delay)
                            else:
                                unlock_date = pd.to_datetime(unlock_date)
                                if unlock_date.tz is not None:
                                    unlock_date = unlock_date.tz_localize(None)
                                
                            # Map unlock_date to trading day index
                            unlock_idx = current_idx + int(self.rights_listing_delay * 5 / 7)
                            for future_idx in range(current_idx, len(self.dates)):
                                target_dt = self.dates[future_idx]
                                if target_dt.tz is not None:
                                    target_dt = target_dt.tz_localize(None)
                                if target_dt >= unlock_date:
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
                                'Note': f"Nháº­n cá»• tá»©c cá»• phiáº¿u tá»‰ lá»‡ {ratio*100:.1f}% (+{new_shares} CP, má»Ÿ khÃ³a ngÃ y {self.dates[min(unlock_idx, len(self.dates)-1)].strftime('%d/%m/%Y')})"
                            })
                            self.order_logs.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'DIVIDEND_STOCK_FILLED',
                                'Reason': f"Nháº­n cá»• tá»©c cá»• phiáº¿u (+{new_shares} CP)",
                                'Price': 0.0,
                                'Quantity': new_shares
                            })

    def _apply_dynamic_rules(self, current_time: pd.Timestamp):
        """Apply VN historical trading rules based on the date."""
        # Quy táº¯c chu ká»³ thanh toÃ¡n lá»‹ch sá»­ Viá»‡t Nam:
        if current_time < pd.Timestamp("2016-01-01"):
            self.settlement_days = 4  # Settle cuá»‘i ngÃ y T+3 -> Giao dá»‹ch tá»« T+4
        elif current_time < pd.Timestamp("2022-08-29"):
            self.settlement_days = 3  # Settle cuá»‘i ngÃ y T+2 -> Giao dá»‹ch tá»« T+3
        else:  # Tá»« 29/08/2022 (T+1.5)
            if self.execution_at == 'close':
                self.settlement_days = 2  # Settle 13:00 T+2 -> Giao dá»‹ch Ä‘Æ°á»£c Close T+2
            else:
                self.settlement_days = 3  # Giao dá»‹ch Ä‘Æ°á»£c Open T+3

    def _detect_if_adjusted(self) -> bool:
        """
        Detect if the input data is already adjusted for splits and dividends.
        We check all historical stock splits/dividends and count how many
        expected price drops are present.
        """
        if not self.adjust_corporate_actions:
            return True
            
        unadjusted_count = 0
        adjusted_count = 0
        
        for ticker, df in self.data.items():
            events = self.corporate_actions.get(ticker)
            if events is None or events.empty:
                continue
                
            # Find stock dividends / splits with a significant ratio (> 0.05)
            splits = events[events['exercise_ratio'] > 0.05]
            df_dates_normalized = df.index.normalize()
            
            for _, event in splits.iterrows():
                if pd.isna(event['exright_date']):
                    continue
                ex_date = pd.to_datetime(event['exright_date']).normalize()
                if ex_date in df_dates_normalized:
                    idx_ex = df_dates_normalized.get_loc(ex_date)
                    if idx_ex > 0:
                        close_prev = df['Close'].iloc[idx_ex - 1]
                        close_ex = df['Close'].iloc[idx_ex]
                        ratio = close_ex / close_prev
                        expected_ratio = 1.0 / (1.0 + event['exercise_ratio'])
                        
                        # Price ratio matches expected price drop (unadjusted)
                        # We allow a wider tolerance of 12% to avoid noise from normal price moves on ex-date
                        if abs(ratio - expected_ratio) < 0.12:
                            unadjusted_count += 1
                        # Price ratio is close to 1.0 (adjusted, meaning no drop)
                        elif abs(ratio - 1.0) < 0.05:
                            adjusted_count += 1
                            
        # If we detected more unadjusted events than adjusted, it's unadjusted
        if unadjusted_count > 0 or adjusted_count > 0:
            print(f"-> PhÃ¢n tÃ­ch dá»¯ liá»‡u: phÃ¡t hiá»‡n {unadjusted_count} sá»± kiá»‡n CHÆ¯A Ä‘iá»u chá»‰nh vÃ  {adjusted_count} sá»± kiá»‡n ÄÃƒ Ä‘iá»u chá»‰nh.")
            return adjusted_count >= unadjusted_count
            
        # Default to True if no events found
        return True

    def _calculate_adjusted_prices(self):
        """
        Calculate adjusted price columns for all tickers.
        If the input price data is already adjusted, we just copy the raw columns.
        If the input data is unadjusted, we use the backward CRSP adjustment algorithm.
        """
        for ticker, df in self.data.items():
            if self.dividends_already_factored:
                # Copy raw columns
                df['Adj_Open'] = df['Open']
                df['Adj_High'] = df['High']
                df['Adj_Low'] = df['Low']
                df['Adj_Close'] = df['Close']
                if 'Average' in df.columns:
                    df['Adj_Average'] = df['Average']
            else:
                # Initialize adjusted columns with raw values
                df['Adj_Open'] = df['Open'].astype(float)
                df['Adj_High'] = df['High'].astype(float)
                df['Adj_Low'] = df['Low'].astype(float)
                df['Adj_Close'] = df['Close'].astype(float)
                if 'Average' in df.columns:
                    df['Adj_Average'] = df['Average'].astype(float)
                    
                events = self.corporate_actions.get(ticker)
                if events is None or events.empty:
                    continue
                    
                events_sorted = events.sort_values('exright_date', ascending=False)
                multipliers = pd.Series(1.0, index=df.index)
                
                for _, event in events_sorted.iterrows():
                    if pd.isna(event['exright_date']):
                        continue
                    ex_date = pd.to_datetime(event['exright_date']).normalize()
                    past_dates = df.index[df.index < ex_date]
                    if past_dates.empty:
                        continue
                        
                    factor = 1.0
                    val_per_share = event.get('value_per_share')
                    exercise_ratio = event.get('exercise_ratio')
                    event_name = event.get('event_name_vi', '')
                    event_title = event.get('event_title_vi', '') if 'event_title_vi' in event.index else ''
                    
                    is_cash_div = False
                    if pd.notna(val_per_share) and val_per_share > 0:
                         is_cash_div = True
                    elif 'tiá»n máº·t' in str(event_name).lower() or 'tiá»n máº·t' in str(event_title).lower():
                         is_cash_div = True
                         if pd.isna(val_per_share) or val_per_share == 0:
                             if pd.notna(exercise_ratio) and exercise_ratio > 0:
                                 val_per_share = exercise_ratio * 10000.0
                             else:
                                 val_per_share = 1000.0
                                 
                    is_rights_issue = False
                    if 'quyá»n mua' in str(event_title).lower() or 'quyá»n mua' in str(event_name).lower():
                        is_rights_issue = True
                                 
                    if is_cash_div:
                         idx_prev_date = past_dates[-1]
                         close_prev = df.loc[idx_prev_date, 'Close']
                         net_div = val_per_share
                         if close_prev > net_div:
                             factor = (close_prev - net_div) / close_prev
                    elif is_rights_issue:
                         ratio = exercise_ratio if pd.notna(exercise_ratio) else 0.0
                         if ratio > 0:
                             idx_prev_date = past_dates[-1]
                             close_prev = df.loc[idx_prev_date, 'Close']
                             subscription_price = 10000.0
                             if close_prev > subscription_price:
                                 factor = (1.0 + ratio * (subscription_price / close_prev)) / (1.0 + ratio)
                             else:
                                 factor = 1.0
                    else:
                         ratio = exercise_ratio if pd.notna(exercise_ratio) else 0.0
                         if ratio > 0:
                             factor = 1.0 / (1.0 + ratio)
                             
                    multipliers.loc[past_dates] *= factor
                    
                df['Adj_Open'] = df['Adj_Open'] * multipliers
                df['Adj_High'] = df['Adj_High'] * multipliers
                df['Adj_Low'] = df['Adj_Low'] * multipliers
                df['Adj_Close'] = df['Adj_Close'] * multipliers
                if 'Average' in df.columns:
                    df['Adj_Average'] = df['Adj_Average'] * multipliers

    def run(self) -> Dict[str, Any]:
        """Run the backtest simulation."""
        # Initialize strategy
        strategy = self.strategy_class(self.data, self, **self.strategy_params)
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

            # 2.6 Auto-liquidation for delisted / last active day stocks
            if hasattr(self, 'last_active_dates'):
                for ticker, last_active_date in self.last_active_dates.items():
                    if current_time.normalize() == last_active_date.normalize():
                        qty = self.positions.get(ticker, 0)
                        if qty > 0:
                            # Force sell at close price of the last active day
                            ticker_df = self.data[ticker]
                            close_price = ticker_df.loc[current_time, 'Close']
                            
                            trade_value = qty * close_price
                            fee = trade_value * self.sell_fee
                            tax = trade_value * self.sell_tax
                            
                            # Apply 5% personal income tax (TNCN) on selling stock dividends under Decree 126
                            div_tax = 0.0
                            div_qty = self.dividend_shares.get(ticker, 0)
                            sold_from_div = min(qty, div_qty)
                            if sold_from_div > 0:
                                self.dividend_shares[ticker] = div_qty - sold_from_div
                                div_tax = sold_from_div * min(close_price, 10000.0) * self.dividend_tax_rate
                                tax += div_tax
                                
                            net_proceeds = trade_value - fee - tax
                            
                            # Add cash and remove positions immediately
                            self.cash += net_proceeds
                            settle_idx = idx + self.settlement_days
                            self.cash_settlement_queue.append({
                                'amount': net_proceeds,
                                'settle_idx': settle_idx,
                                'borrowed': 0.0
                            })
                            
                            self.positions[ticker] = 0
                            self.sellable_shares[ticker] = 0
                            
                            # Clean up positions dictionaries
                            if ticker in self.positions:
                                del self.positions[ticker]
                            if ticker in self.sellable_shares:
                                del self.sellable_shares[ticker]
                            if ticker in self.position_entry_price:
                                del self.position_entry_price[ticker]
                            if ticker in self.position_highest_price:
                                del self.position_highest_price[ticker]
                            if ticker in self.dividend_shares:
                                del self.dividend_shares[ticker]
                                
                            self.trades_history.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'SELL',
                                'Quantity': qty,
                                'Price': close_price,
                                'Value': trade_value,
                                'Fee': fee,
                                'Tax': tax,
                                'TotalValue': net_proceeds,
                                'TimePlaced': current_time,
                                'Note': 'Auto-liquidated on last active trading day (Delisted)'
                            })
                            self.order_logs.append({
                                'Date': current_time,
                                'Ticker': ticker,
                                'Action': 'SELL_FILLED',
                                'Reason': 'Auto-liquidated on last active trading day (Delisted)',
                                'Price': close_price,
                                'Quantity': qty
                            })
                            print(f"   [DELIST] Tá»± Ä‘á»™ng táº¥t toÃ¡n vá»‹ tháº¿ {ticker}: BÃ¡n {qty} CP táº¡i giÃ¡ {close_price:,.0f}Ä‘ do há»§y niÃªm yáº¿t.")

            # Äá»“ng bá»™ hÃ³a sá»©c mua kháº£ dá»¥ng trÃ¡nh bá»‹ vÆ°á»£t quÃ¡ lÆ°á»£ng tiá»n máº·t thá»±c táº¿ (trá»« trÆ°á»ng há»£p dÃ¹ng Margin)
            if self.margin_ratio >= 1.0:
                self.available_cash = max(0.0, min(self.available_cash, self.cash))

            # 2.7 Auto Stop Loss & Trailing Stop Risk checks
            self._check_risk_management(strategy, current_time, idx)

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
                    if exch == 'upcom':
                        if 'Average' in prev_row and pd.notna(prev_row['Average']):
                            prev_closes[ticker] = prev_row['Average']
                        else:
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
            # Calculate settled cash (excluding pending cash from sells that hasn't settled yet)
            pending_cash = sum(item['amount'] - item.get('borrowed', 0.0) for item in self.cash_settlement_queue)
            settled_cash = self.cash - pending_cash
            
            if settled_cash < 0:
                # Calculate actual calendar days elapsed since the previous trading day
                days_diff = 1
                if idx > 0:
                    days_diff = (self.dates[idx] - self.dates[idx-1]).days
                    if days_diff <= 0:
                        days_diff = 1
                interest = abs(settled_cash) * (self.margin_interest_rate / 365.0) * days_diff
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
                    'Note': f"LÃ£i vay Margin ({days_diff} ngÃ y): {interest:,.0f} VND (DÆ° ná»£ thá»±c táº¿: {abs(settled_cash):,.0f} VND, Chá» vá»: {pending_cash:,.0f} VND)"
                })
                
            # 4.2 Margin Maintenance Ratio Check (Force Sell liquidation)
            if positions_value > 0 and self.margin_ratio < 1.0:
                current_margin_ratio = equity / positions_value
                if current_margin_ratio < self.margin_maintenance_ratio:
                    # Calculate required liquidation value
                    target_ratio = self.margin_maintenance_ratio + 0.02
                    value_to_sell = (target_ratio * positions_value - equity) / (target_ratio - self.sell_fee - self.sell_tax)
                    value_to_sell = min(value_to_sell, positions_value)
                    
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': 'PORTFOLIO',
                        'Action': 'MARGIN_CALL',
                        'Reason': f"Tá»· lá»‡ kÃ½ quá»¹ ({current_margin_ratio*100:.2f}%) < {self.margin_maintenance_ratio*100:.2f}%. YÃªu cáº§u giáº£i cháº¥p khoáº£ng {value_to_sell:,.0f} VND.",
                        'Price': 0.0,
                        'Quantity': 0
                    })
                    
                    for ticker, qty in list(self.positions.items()):
                        # Proportional sell
                        ticker_df = self.data[ticker]
                        if current_time in ticker_df.index:
                            close_price = ticker_df.loc[current_time, 'Close']
                        else:
                            past_df = ticker_df[:current_time]
                            close_price = past_df.iloc[-1]['Close'] if not past_df.empty else 0.0
                            
                        if close_price > 0:
                            qty_to_sell = qty * (value_to_sell / positions_value)
                            lot_size = self._get_lot_size(ticker, current_time)
                            effective_lot_size = 1 if self._is_odd_lot_allowed(ticker, current_time) else lot_size
                            if effective_lot_size and effective_lot_size > 0:
                                qty_to_sell = int(np.ceil(qty_to_sell / effective_lot_size)) * effective_lot_size
                            else:
                                qty_to_sell = int(np.ceil(qty_to_sell))
                                
                            qty_to_sell = min(qty_to_sell, qty)
                            if qty_to_sell > 0:
                                self.place_sell_order(ticker, size=qty_to_sell, time=current_time)
            
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
                self._size_pending_orders(current_time, idx)
                
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
                
                # Apply 5% personal income tax (TNCN) on selling stock dividends under Decree 126
                div_tax = 0.0
                div_qty = self.dividend_shares.get(ticker, 0)
                sold_from_div = min(qty, div_qty)
                if sold_from_div > 0:
                    self.dividend_shares[ticker] = div_qty - sold_from_div
                    div_tax = sold_from_div * min(close_price, 10000.0) * self.dividend_tax_rate
                    tax += div_tax
                    
                net_proceeds = trade_value - fee - tax
                
                # Settle cash immediately since it's the end of backtest
                self.cash += net_proceeds
                self.available_cash += net_proceeds
                
                self.positions[ticker] = 0
                self.sellable_shares[ticker] = 0
                if ticker in self.dividend_shares:
                    del self.dividend_shares[ticker]
                
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
        """Execute orders queued on the previous bar. Uses Order objects for full lifecycle."""
        if not self.pending_orders:
            return

        orders_to_process = self.pending_orders.copy()
        self.pending_orders.clear()

        # --- Phase 1: Tick all orders and handle expiry ---
        active_orders = []
        for order in orders_to_process:
            if order.tick():
                # Order has expired
                self.order_logs.append({
                    'Date': current_time, 'Ticker': order.ticker,
                    'Action': f'{order.action.upper()}_EXPIRED',
                    'Reason': f'Order #{order.order_id} expired after {order.bars_alive} bars (type={order.order_type.value})',
                    'Price': order.limit_price or order.stop_price or 0.0,
                    'Quantity': order.remaining_quantity
                })
            else:
                active_orders.append(order)

        # --- Phase 2: Check Stop triggers ---
        for order in active_orders:
            if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                ticker_df = self.data.get(order.ticker)
                if ticker_df is not None and current_time in ticker_df.index:
                    row = ticker_df.loc[current_time]
                    if order.check_stop_trigger(float(row['High']), float(row['Low'])):
                        # Convert: STOP â†’ MARKET, STOP_LIMIT â†’ LIMIT
                        original_type = order.order_type.value
                        if order.order_type == OrderType.STOP:
                            order.order_type = OrderType.MARKET
                        elif order.order_type == OrderType.STOP_LIMIT:
                            order.order_type = OrderType.LIMIT
                        self.order_logs.append({
                            'Date': current_time, 'Ticker': order.ticker,
                            'Action': f'{order.action.upper()}_STOP_TRIGGERED',
                            'Reason': f'Order #{order.order_id} {original_type} triggered at stop={order.stop_price:,.0f} â†’ {order.order_type.value}',
                            'Price': order.stop_price,
                            'Quantity': order.remaining_quantity
                        })

        # Sort: SELL first, BUY second to free up cash before buying
        sorted_orders = sorted(active_orders, key=lambda x: 0 if x.action == 'sell' else 1)

        # --- Phase 3: Execute each order ---
        for order in sorted_orders:
            # Skip if cancelled by OCO in this loop iteration
            if not order.is_active:
                continue

            # Skip un-triggered stop orders (keep in pending for next bar)
            if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not order._stop_triggered:
                self.pending_orders.append(order)
                continue

            ticker = order.ticker
            action = order.action
            qty = order.remaining_quantity
            time_placed = order.time_placed

            # Check if ticker traded on current_time
            ticker_df = self.data[ticker]
            if current_time not in ticker_df.index:
                self.pending_orders.append(order)
                continue

            row = ticker_df.loc[current_time]
            # Check for zero volume or NaN volume (suspended trading or illiquid) or not traded today
            if ('Traded' in row and row['Traded'] == 0) or ('Volume' in row and (pd.isna(row['Volume']) or row['Volume'] <= 0)):
                self.pending_orders.append(order)
                self.order_logs.append({
                    'Date': current_time, 'Ticker': ticker,
                    'Action': f'{action.upper()}_DEFERRED',
                    'Reason': f'Order #{order.order_id}: Zero trading volume or suspended trading',
                    'Price': 0.0, 'Quantity': 0
                })
                continue

            prev_close = prev_closes.get(ticker)

            # Check if today is the listing day
            is_listing_day = False
            if hasattr(self, 'raw_listing_dates') and ticker in self.raw_listing_dates:
                if current_time.normalize() == self.raw_listing_dates[ticker].normalize():
                    is_listing_day = True

            if is_listing_day:
                prev_close = float(row['Open'])

            exch = self.exchanges.get(ticker, "hose")
            lot_size = self._get_lot_size(ticker, current_time)
            price_limit = self._get_price_limit(ticker, current_time)
            effective_lot_size = 1 if self._is_odd_lot_allowed(ticker, current_time) else lot_size

            # --- Determine execution price ---
            limit_price = order.limit_price

            # Handle triggered stop orders: use stop_price as reference
            if order._stop_triggered and limit_price is None:
                # Pure STOP â†’ now MARKET: fill at stop price or Open if gapped through
                if action == 'sell':
                    base_price = min(float(row['Open']), order.stop_price)
                else:
                    base_price = max(float(row['Open']), order.stop_price)
            elif limit_price is not None:
                # Limit order (or Stop-Limit after trigger)
                if action == 'buy':
                    if row['Low'] <= limit_price:
                        base_price = min(float(row['Open']), limit_price)
                    else:
                        self.pending_orders.append(order)
                        continue
                elif action == 'sell':
                    if row['High'] >= limit_price:
                        base_price = max(float(row['Open']), limit_price)
                    else:
                        self.pending_orders.append(order)
                        continue
                else:
                    base_price = self._get_execution_price(row)
            else:
                base_price = self._get_execution_price(row)

            # Apply Price Limits (Ceiling/Floor)
            ceiling, floor, is_ceiling, is_floor = self._check_price_limits(base_price, prev_close, exch, price_limit, current_time)

            # Check Ceiling/Floor Locks
            if action == 'buy' and is_ceiling and self.restrict_ceiling_buy:
                self.order_logs.append({
                    'Date': current_time, 'Ticker': ticker,
                    'Action': 'BUY_REJECTED', 'Reason': f'Order #{order.order_id}: Price at Ceiling limit ({ceiling})',
                    'Price': base_price, 'Quantity': 0
                })
                continue

            if action == 'sell' and is_floor and self.restrict_floor_sell:
                self.order_logs.append({
                    'Date': current_time, 'Ticker': ticker,
                    'Action': 'SELL_REJECTED', 'Reason': f'Order #{order.order_id}: Price at Floor limit ({floor})',
                    'Price': base_price, 'Quantity': 0
                })
                continue

            # --- Apply Slippage (only for market orders and triggered stops, not limit orders) ---
            exec_price = base_price
            if limit_price is None:
                pct_slippage = self.slippage
                if self.market_impact_coef > 0.0 and 'Volume' in row and row['Volume'] > 0:
                    volume_share = qty / row['Volume']
                    pct_slippage += self.market_impact_coef * (volume_share ** 2)

                pct_slippage = min(pct_slippage, 0.05)

                if action == 'buy':
                    exec_price = exec_price * (1.0 + pct_slippage)
                else:
                    exec_price = exec_price * (1.0 - pct_slippage)

                order.applied_slippage = pct_slippage

                tick_size = self._get_tick_size(base_price, exch, current_time)
                min_slippage = 0.5 * tick_size

                if action == 'buy':
                    exec_price += min_slippage
                    exec_price = self._round_to_tick(exec_price, exch, "up", current_time)
                else:
                    exec_price -= min_slippage
                    exec_price = self._round_to_tick(exec_price, exch, "down", current_time)
            else:
                exec_price = self._round_to_tick(exec_price, exch, "nearest", current_time)

            if exec_price > ceiling:
                exec_price = ceiling
            elif exec_price < floor:
                exec_price = floor

            # ============================================================
            # --- PROCESS BUY ORDER ---
            # ============================================================
            if action == 'buy':
                original_qty = qty
                # Apply volume limit constraint
                if self.max_volume_ratio is not None and 'Volume' in row:
                    max_qty = int(row['Volume'] * self.max_volume_ratio)
                    if effective_lot_size and effective_lot_size > 0:
                        max_qty = int(max_qty // effective_lot_size) * effective_lot_size
                    if qty > max_qty:
                        qty = max_qty

                deferred_qty = original_qty - qty

                if qty <= 0:
                    if self.partial_fill_mode == 'defer':
                        self.pending_orders.append(order)
                        self.order_logs.append({
                            'Date': current_time, 'Ticker': ticker,
                            'Action': 'BUY_DEFERRED',
                            'Reason': f'Order #{order.order_id}: Target quantity scaled to 0 due to volume limits (deferred {original_qty} shares)',
                            'Price': exec_price, 'Quantity': original_qty
                        })
                    else:
                        order.cancel()
                        self.order_logs.append({
                            'Date': current_time, 'Ticker': ticker,
                            'Action': 'BUY_CANCELLED',
                            'Reason': f'Order #{order.order_id}: Insufficient liquidity (partial_fill_mode=cancel)',
                            'Price': exec_price, 'Quantity': original_qty
                        })
                    continue

                if deferred_qty > 0:
                    self.order_logs.append({
                        'Date': current_time, 'Ticker': ticker,
                        'Action': 'BUY_PARTIALLY_DEFERRED',
                        'Reason': f'Order #{order.order_id}: Filling {qty} shares, deferred {deferred_qty} due to volume limits',
                        'Price': exec_price, 'Quantity': deferred_qty
                    })

                # Calculate Net Equity and max spend for margin trading
                current_positions_value = sum(
                    q * prev_closes.get(t, 0.0)
                    for t, q in self.positions.items()
                    if prev_closes.get(t) is not None
                )
                net_equity = self.cash + current_positions_value
                max_leverage = 1.0 / self.margin_ratio
                max_spend = max(0.0, net_equity * max_leverage - current_positions_value)

                # Calculate cost
                trade_value = qty * exec_price
                fee = trade_value * self.buy_fee
                total_cost = trade_value + fee

                amount_needed = total_cost - self.available_cash
                advance_fee = 0.0

                # If we exceed max spend, scale down qty via binary search
                if total_cost > max_spend:
                    low_qty = 0
                    high_qty = qty
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
                            'Date': current_time, 'Ticker': ticker,
                            'Action': 'BUY_CANCELLED',
                            'Reason': f'Order #{order.order_id}: Insufficient funds including pending cash',
                            'Price': exec_price, 'Quantity': 0
                        })
                        order.cancel()
                        continue

                    trade_value = qty * exec_price
                    fee = trade_value * self.buy_fee
                    total_cost = trade_value + fee
                    amount_needed = total_cost - self.available_cash

                # Borrow from cash_settlement_queue if needed
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
                    'TimePlaced': time_placed,
                    'OrderID': order.order_id,
                    'OrderType': order.order_type.value,
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': current_time, 'Ticker': ticker,
                    'Action': 'BUY_FILLED',
                    'Reason': f'Order #{order.order_id} {order.order_type.value}: Success' + (f' (á»¨ng trÆ°á»›c, phÃ­: {advance_fee:,.0f}Ä‘)' if advance_fee > 0 else ''),
                    'Price': exec_price, 'Quantity': qty
                })

                # Record fill on the Order object
                order.fill(qty, exec_price)

                # Update tracking of entry prices
                old_qty = self.positions.get(ticker, 0) - qty
                if old_qty > 0:
                    old_price = self.position_entry_price.get(ticker, exec_price)
                    self.position_entry_price[ticker] = (old_price * old_qty + exec_price * qty) / (old_qty + qty)
                else:
                    self.position_entry_price[ticker] = exec_price

                if self.execution_at == 'open':
                    self.position_highest_price[ticker] = max(self.position_highest_price.get(ticker, 0.0), exec_price, row['High'])
                else:
                    self.position_highest_price[ticker] = max(self.position_highest_price.get(ticker, 0.0), exec_price)

                # If order still has remaining quantity (partial fill), keep in pending
                if order.is_active and order.remaining_quantity > 0:
                    if self.partial_fill_mode == 'defer':
                        self.pending_orders.append(order)
                    else:
                        order.cancel()

            # ============================================================
            # --- PROCESS SELL ORDER ---
            # ============================================================
            elif action == 'sell':
                # Get max sellable quantity
                max_sellable = self.sellable_shares.get(ticker, 0)
                original_qty = qty
                qty = min(qty, max_sellable)

                # Check if we need to defer the unsellable part
                deferred_qty = original_qty - qty

                if qty <= 0:
                    if self.partial_fill_mode == 'defer':
                        self.pending_orders.append(order)
                        self.order_logs.append({
                            'Date': current_time, 'Ticker': ticker,
                            'Action': 'SELL_DEFERRED',
                            'Reason': f'Order #{order.order_id}: All shares locked in settlement (deferred {original_qty} shares)',
                            'Price': exec_price, 'Quantity': original_qty
                        })
                    else:
                        order.cancel()
                        self.order_logs.append({
                            'Date': current_time, 'Ticker': ticker,
                            'Action': 'SELL_CANCELLED',
                            'Reason': f'Order #{order.order_id}: All shares locked (partial_fill_mode=cancel)',
                            'Price': exec_price, 'Quantity': original_qty
                        })
                    continue

                # Apply volume limit constraint if specified
                if self.max_volume_ratio is not None and 'Volume' in row:
                    max_qty = int(row['Volume'] * self.max_volume_ratio)
                    if effective_lot_size and effective_lot_size > 0:
                        max_qty = int(max_qty // effective_lot_size) * effective_lot_size
                    if qty > max_qty:
                        qty = max_qty
                        deferred_qty = original_qty - qty

                if qty <= 0:
                    if self.partial_fill_mode == 'defer':
                        self.pending_orders.append(order)
                    else:
                        order.cancel()
                    self.order_logs.append({
                        'Date': current_time, 'Ticker': ticker,
                        'Action': 'SELL_CANCELLED',
                        'Reason': f'Order #{order.order_id}: Target quantity scaled to 0 due to volume limits',
                        'Price': exec_price, 'Quantity': 0
                    })
                    continue

                if deferred_qty > 0:
                    self.order_logs.append({
                        'Date': current_time, 'Ticker': ticker,
                        'Action': 'SELL_PARTIALLY_DEFERRED',
                        'Reason': f'Order #{order.order_id}: Sold {qty} shares, deferred {deferred_qty} due to settlement lock / volume limits',
                        'Price': exec_price, 'Quantity': deferred_qty
                    })

                # Execute Sell Trade
                trade_value = qty * exec_price
                fee = trade_value * self.sell_fee
                tax = trade_value * self.sell_tax

                # Apply 5% personal income tax (TNCN) on selling stock dividends under Decree 126
                div_tax = 0.0
                div_qty = self.dividend_shares.get(ticker, 0)
                sold_from_div = min(qty, div_qty)
                if sold_from_div > 0:
                    self.dividend_shares[ticker] = div_qty - sold_from_div
                    div_tax = sold_from_div * min(exec_price, 10000.0) * self.dividend_tax_rate
                    tax += div_tax

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
                    if ticker in self.position_entry_price:
                        del self.position_entry_price[ticker]
                    if ticker in self.position_highest_price:
                        del self.position_highest_price[ticker]
                    if ticker in self.dividend_shares:
                        del self.dividend_shares[ticker]
                if ticker in self.sellable_shares and self.sellable_shares[ticker] == 0:
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
                    'TimePlaced': time_placed,
                    'OrderID': order.order_id,
                    'OrderType': order.order_type.value,
                }
                self.trades_history.append(trade_record)
                self.order_logs.append({
                    'Date': current_time, 'Ticker': ticker,
                    'Action': 'SELL_FILLED',
                    'Reason': f'Order #{order.order_id} {order.order_type.value}: Success',
                    'Price': exec_price, 'Quantity': qty
                })

                # Record fill on the Order object
                order.fill(qty, exec_price)

                # If order still has remaining quantity, keep in pending
                if order.is_active and order.remaining_quantity > 0:
                    if self.partial_fill_mode == 'defer':
                        self.pending_orders.append(order)
                    else:
                        order.cancel()

            # --- Phase 4: Handle OCO group after each fill ---
            if order.filled_quantity > 0 and order.oco_group_id is not None:
                group = self.oco_groups.get(order.oco_group_id)
                if group is not None:
                    cancelled_ids = group.on_fill(order.order_id, self.all_orders)
                    if cancelled_ids:
                        # Remove cancelled OCO siblings from pending_orders
                        cancelled_set = set(cancelled_ids)
                        self.pending_orders = [o for o in self.pending_orders if o.order_id not in cancelled_set]
                        for cid in cancelled_ids:
                            c_order = self.all_orders.get(cid)
                            self.order_logs.append({
                                'Date': current_time,
                                'Ticker': c_order.ticker if c_order else ticker,
                                'Action': f'{c_order.action.upper()}_CANCELLED_OCO' if c_order else 'CANCELLED_OCO',
                                'Reason': f'Order #{cid} cancelled by OCO group #{order.oco_group_id} (sibling #{order.order_id} filled)',
                                'Price': 0.0, 'Quantity': c_order.remaining_quantity if c_order else 0
                            })

    def _reindex_and_fill_data(self):
        """Reindex each ticker's DataFrame to the unified timeline and fill missing values."""
        self.last_active_dates = {}
        for ticker, df in list(self.data.items()):
            df['Traded'] = 1.0
            
            # Keep track of the actual last active trading date in the raw data
            last_active_date = df.index[-1]
            self.last_active_dates[ticker] = last_active_date
            
            # Reindex to the global timeline
            df_reindexed = df.reindex(self.dates)
            
            # Fill Volume and Traded with 0 for missing days
            df_reindexed['Volume'] = df_reindexed['Volume'].fillna(0.0)
            df_reindexed['Traded'] = df_reindexed['Traded'].fillna(0.0)
            
            # Identify which price columns actually exist in the original DataFrame
            price_cols = [
                'Open', 'High', 'Low', 'Close', 
                'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close',
                'Average', 'Adj_Average'
            ]
            actual_price_cols = [col for col in price_cols if col in df_reindexed.columns]
            
            # Forward fill price columns, but do NOT backward fill (to prevent lookahead/listing bias before debut)
            df_reindexed[actual_price_cols] = df_reindexed[actual_price_cols].ffill()
            
            # Set price columns to NaN after the last active date to prevent look-ahead bias and hold pricing
            df_reindexed.loc[df_reindexed.index > last_active_date, actual_price_cols] = np.nan
            
            self.data[ticker] = df_reindexed

    def _size_pending_orders(self, current_time: pd.Timestamp, current_idx: int):
        """
        Convert float/None sizes and target percent orders into exact share quantities
        using the close prices of the current day (at the time the order is placed).
        Works with Order objects — updates them in place.
        """
        sized_orders = []
        for order in self.pending_orders:
            action = order.action
            ticker = order.ticker

            # Check if this order has already been sized (e.g. deferred from a previous day due to settlement lock)
            if order.is_sized:
                # Keep it as is but cap the sell quantity if our total position has changed
                if action == 'sell':
                    total_pos = self.positions.get(ticker, 0)
                    order.quantity = min(order.remaining_quantity, total_pos)
                    if order.quantity <= 0:
                        continue
                sized_orders.append(order)
                continue

            # Get latest close price of the ticker
            ticker_df = self.data[ticker]
            if current_time in ticker_df.index:
                close_price = ticker_df.loc[current_time, 'Close']
            else:
                past_df = ticker_df[:current_time]
                close_price = past_df.iloc[-1]['Close'] if not past_df.empty else None

            if close_price is None or pd.isna(close_price) or close_price <= 0:
                # Cannot size order without price, reject it
                self.order_logs.append({
                    'Date': current_time,
                    'Ticker': ticker,
                    'Action': f'{action.upper()}_REJECTED',
                    'Reason': f'Order #{order.order_id}: No historical price available for order sizing',
                    'Price': 0.0,
                    'Quantity': 0
                })
                order.cancel()
                continue

            lot_size = self._get_lot_size(ticker, current_time)
            effective_lot_size = 1 if self._is_odd_lot_allowed(ticker, current_time) else lot_size

            # Calculate total portfolio equity for target percent or margin calculations
            positions_value = 0.0
            for t, q in self.positions.items():
                t_df = self.data[t]
                if current_time in t_df.index:
                    c_price = t_df.loc[current_time, 'Close']
                else:
                    past_t = t_df[:current_time]
                    c_price = past_t.iloc[-1]['Close'] if not past_t.empty else 0.0
                positions_value += q * c_price
            equity = self.cash + positions_value

            qty = 0
            if action == 'target_percent':
                target_percent = order.target_percent
                current_qty = self.positions.get(ticker, 0)

                if target_percent == 0.0:
                    # Sell all positions (using total position size to support T+2 deferral)
                    qty = self.positions.get(ticker, 0)
                    action = 'sell'
                else:
                    target_value = equity * target_percent
                    current_value = current_qty * close_price

                    if target_value > current_value:
                        # Need to buy
                        cash_to_use = target_value - current_value
                        target_shares = cash_to_use / (close_price * (1 + self.buy_fee))
                        if effective_lot_size and effective_lot_size > 0:
                            qty = int(target_shares // effective_lot_size) * effective_lot_size
                        else:
                            qty = int(target_shares)
                        action = 'buy'
                    elif target_value < current_value:
                        # Need to sell
                        value_to_sell = current_value - target_value
                        target_shares = value_to_sell / close_price
                        if effective_lot_size and effective_lot_size > 0:
                            qty = int(target_shares // effective_lot_size) * effective_lot_size
                        else:
                            qty = int(target_shares)
                        total_pos = self.positions.get(ticker, 0)
                        qty = min(qty, total_pos)
                        action = 'sell'
                    else:
                        continue  # No change needed
            else:
                # Standard buy or sell
                size = order.size
                if action == 'buy':
                    if size is None:
                        # Use all available cash (or max spend if margin)
                        if self.margin_ratio < 1.0:
                            max_leverage = 1.0 / self.margin_ratio
                            max_spend = max(0.0, equity * max_leverage - positions_value)
                            cash_to_use = max_spend
                        else:
                            cash_to_use = self.available_cash
                    elif isinstance(size, float) and 0.0 < size <= 1.0:
                        cash_to_use = equity * size
                        if self.margin_ratio < 1.0:
                            max_leverage = 1.0 / self.margin_ratio
                            max_spend = max(0.0, equity * max_leverage - positions_value)
                            cash_to_use = min(cash_to_use, max_spend)
                        else:
                            cash_to_use = min(cash_to_use, self.available_cash)
                    elif isinstance(size, (int, np.integer)) and size >= 1:
                        qty = int(size)
                        cash_to_use = 0.0
                    else:
                        continue

                    if qty == 0:
                        target_shares = cash_to_use / (close_price * (1 + self.buy_fee))
                        if effective_lot_size and effective_lot_size > 0:
                            qty = int(target_shares // effective_lot_size) * effective_lot_size
                        else:
                            qty = int(target_shares)

                elif action == 'sell':
                    # Size against total positions to support T+2 deferral
                    total_pos = self.positions.get(ticker, 0)
                    if size is None:
                        qty = total_pos
                    elif isinstance(size, float) and 0.0 < size <= 1.0:
                        target_shares = total_pos * size
                        if effective_lot_size and effective_lot_size > 0:
                            qty = int(target_shares // effective_lot_size) * effective_lot_size
                            if qty == 0 and target_shares > 0 and target_shares == total_pos:
                                qty = total_pos
                        else:
                            qty = int(target_shares)
                    elif isinstance(size, (int, np.integer)) and size >= 1:
                        qty = min(int(size), total_pos)
                    else:
                        continue

            if qty > 0:
                # Update the Order object in place
                order.action = action
                order.quantity = qty
                order.is_sized = True
                sized_orders.append(order)

        self.pending_orders = sized_orders

    def _check_risk_management(self, strategy, current_time: pd.Timestamp, current_idx: int):
        """Check and execute Stop Loss / Trailing Stop triggers for active positions."""
        if not self.positions:
            return
            
        has_sl = getattr(strategy, 'stop_loss', None) is not None
        has_ts = getattr(strategy, 'trailing_stop', None) is not None
        
        if not (has_sl or has_ts):
            return
            
        tickers_to_check = list(self.positions.keys())
        for ticker in tickers_to_check:
            qty = self.positions.get(ticker, 0)
            if qty <= 0:
                continue
                
            ticker_df = self.data[ticker]
            if current_time not in ticker_df.index:
                continue
                
            row = ticker_df.loc[current_time]
            # Skip checking if not traded today
            if 'Traded' in row and row['Traded'] == 0:
                continue
                
            low_price = row['Low']
            high_price = row['High']
            open_price = row['Open']
            
            # Get position highest price before today to calculate trailing stop level
            prev_highest = self.position_highest_price.get(ticker, open_price)
            
            # Check Stop Loss
            triggered = False
            trigger_reason = ""
            trigger_price = 0.0
            
            if has_sl:
                entry_price = self.position_entry_price.get(ticker, open_price)
                stop_loss_price = entry_price * (1.0 - strategy.stop_loss)
                if low_price <= stop_loss_price:
                    triggered = True
                    trigger_reason = f"Stop Loss Triggered (-{strategy.stop_loss*100:.1f}%)"
                    trigger_price = min(open_price, stop_loss_price) # Sell at stop price or Open if gap down
                    
            # Check Trailing Stop
            if not triggered and has_ts:
                # Use highest price before today to calculate trailing stop level
                # to avoid today's high resetting the trailing stop before checking today's low.
                trailing_stop_price = prev_highest * (1.0 - strategy.trailing_stop)
                if low_price <= trailing_stop_price:
                    triggered = True
                    trigger_reason = f"Trailing Stop Triggered (-{strategy.trailing_stop*100:.1f}%)"
                    trigger_price = min(open_price, trailing_stop_price) # Sell at trailing stop price or Open if gap down
            
            if triggered:
                # Execute Sell immediately (at trigger_price)
                sellable = self.sellable_shares.get(ticker, 0)
                
                # Check if we need to defer the unsellable part
                deferred_qty = qty - sellable
                
                if sellable <= 0:
                    # Cannot sell due to settlement lock, queue a pending sell order for next day
                    self.place_sell_order(ticker, size=qty, time=current_time)
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'RISK_TRIGGER_DEFERRED',
                        'Reason': f"{trigger_reason} but shares are locked in settlement. Queued sell order.",
                        'Price': trigger_price,
                        'Quantity': qty
                    })
                else:
                    qty_to_sell = min(qty, sellable)
                    
                    if deferred_qty > 0:
                        # Queue the remaining unsellable portion for next day
                        self.place_sell_order(ticker, size=deferred_qty, time=current_time)
                        self.order_logs.append({
                            'Date': current_time,
                            'Ticker': ticker,
                            'Action': 'RISK_TRIGGER_PARTIALLY_DEFERRED',
                            'Reason': f"{trigger_reason} but partial lock. Sold {qty_to_sell}, queued remaining {deferred_qty}.",
                            'Price': trigger_price,
                            'Quantity': deferred_qty
                        })
                        
                    trade_value = qty_to_sell * trigger_price
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
                    
                    self.positions[ticker] = self.positions[ticker] - qty_to_sell
                    self.sellable_shares[ticker] = self.sellable_shares[ticker] - qty_to_sell
                    
                    if self.positions[ticker] == 0:
                        del self.positions[ticker]
                        if ticker in self.position_entry_price:
                            del self.position_entry_price[ticker]
                        if ticker in self.position_highest_price:
                            del self.position_highest_price[ticker]
                    
                    if ticker in self.sellable_shares and self.sellable_shares[ticker] == 0:
                        del self.sellable_shares[ticker]
                        
                    trade_record = {
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'SELL',
                        'Quantity': qty_to_sell,
                        'Price': trigger_price,
                        'Value': trade_value,
                        'Fee': fee,
                        'Tax': tax,
                        'TotalValue': net_proceeds,
                        'TimePlaced': current_time,
                        'Note': trigger_reason
                    }
                    self.trades_history.append(trade_record)
                    self.order_logs.append({
                        'Date': current_time,
                        'Ticker': ticker,
                        'Action': 'SELL_FILLED',
                        'Reason': trigger_reason,
                        'Price': trigger_price,
                        'Quantity': qty_to_sell
                    })
            
            # Update position highest price at the end of the day if the position still exists
            if ticker in self.positions:
                self.position_highest_price[ticker] = max(prev_highest, high_price)
