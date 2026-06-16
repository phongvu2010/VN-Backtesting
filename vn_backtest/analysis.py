import pandas as pd
import numpy as np
from typing import Dict, Any

class PerformanceAnalyzer:
    """
    Computes key performance metrics for trading strategies,
    comparing the results with the VN-Index benchmark.
    """
    @staticmethod
    def calculate_metrics(
        equity_curve: pd.DataFrame, 
        trades: pd.DataFrame, 
        benchmark_data: pd.DataFrame = None,
        initial_cash: float = 100_000_000.0,
        risk_free_rate: float = 0.04,  # 4% risk-free rate typical in VN
        include_auto_close: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate metrics.
        
        Args:
            equity_curve (pd.DataFrame): DataFrame with index 'Date' and column 'Equity'.
            trades (pd.DataFrame): DataFrame of executed trades.
            benchmark_data (pd.DataFrame, optional): DataFrame with index 'Date' and column 'Close' (benchmark).
            initial_cash (float): Starting portfolio value.
            risk_free_rate (float): Annualized risk-free rate.
            
        Returns:
            Dict: Financial metrics.
        """
        if equity_curve.empty:
            return {}

        final_equity = equity_curve['Equity'].iloc[-1]
        total_return = (final_equity - initial_cash) / initial_cash
        
        # Calculate calendar duration
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        duration_days = (end_date - start_date).days
        years = duration_days / 365.25
        
        # CAGR
        if final_equity <= 0:
            cagr = -1.0
        elif years > 0:
            cagr = (final_equity / initial_cash) ** (1 / years) - 1
        else:
            cagr = 0.0

        # Daily Returns
        equity_curve['DailyReturn'] = equity_curve['Equity'].pct_change().fillna(0.0)
        daily_returns = equity_curve['DailyReturn']

        # Volatility (Annualized)
        daily_vol = daily_returns.std()
        ann_vol = daily_vol * np.sqrt(252)

        # Sharpe Ratio
        if ann_vol > 0:
            sharpe_ratio = (cagr - risk_free_rate) / ann_vol
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio
        # downside deviation: replace positive returns with 0
        downside_diff = np.minimum(daily_returns, 0.0)
        downside_vol = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
        if downside_vol > 0:
            sortino_ratio = (cagr - risk_free_rate) / downside_vol
        else:
            sortino_ratio = 0.0

        # Drawdowns
        running_max = equity_curve['Equity'].cummax()
        drawdown = (equity_curve['Equity'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Drawdown Duration (in trading days)
        is_in_drawdown = drawdown < 0
        drawdown_streaks = is_in_drawdown.groupby((~is_in_drawdown).cumsum()).cumsum()
        max_dd_duration = int(drawdown_streaks.max()) if not drawdown_streaks.empty else 0

        # Trade Statistics
        # Filter out auto-closed trades from trade-level stats to avoid skewing win rate/profit factor if requested
        strategy_trades = trades
        if not include_auto_close and not trades.empty and 'Note' in trades.columns:
            strategy_trades = trades[trades['Note'].isna() | (trades['Note'] != 'Auto-closed at end of backtest')]

        # Count only BUY and SELL for trade count statistics
        actual_buy_sells = pd.DataFrame(columns=trades.columns)
        if not strategy_trades.empty:
            actual_buy_sells = strategy_trades[strategy_trades['Action'].isin(['BUY', 'SELL'])]
        total_trades = len(actual_buy_sells)
        
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        avg_hold_days = 0.0

        if total_trades > 0:
            # Filter for trade matching (BUY, SELL, DIVIDEND_STOCK, and DIVIDEND_CASH)
            matching_trades = pd.DataFrame(columns=trades.columns)
            if not strategy_trades.empty:
                matching_trades = strategy_trades[strategy_trades['Action'].isin(['BUY', 'SELL', 'DIVIDEND_STOCK', 'DIVIDEND_CASH'])]
            
            # We pair BUYs and SELLs to calculate individual trade profits.
            # In simple portfolio trading, a trade starts with a BUY and ends with a SELL.
            # Let's match trades by FIFO per ticker.
            completed_trades = []
            buy_queues = {}
            
            # Sort trades chronologically
            trades_sorted = matching_trades.sort_values('Date')
            
            for _, t in trades_sorted.iterrows():
                ticker = t['Ticker']
                if ticker not in buy_queues:
                    buy_queues[ticker] = []

                if t['Action'] == 'BUY':
                    # Add buying lot
                    # Include cash advance fee in the buy fee if it was incurred
                    advance_fee = t['AdvanceFee'] if 'AdvanceFee' in t.index and pd.notna(t['AdvanceFee']) else 0.0
                    buy_queues[ticker].append({
                        'qty': t['Quantity'],
                        'price': t['Price'],
                        'date': t['Date'],
                        'fee': t['Fee'] + advance_fee
                    })
                elif t['Action'] == 'DIVIDEND_STOCK':
                    # Adjust the cost basis of all existing lots proportionally instead of adding a 0-cost lot
                    total_qty_before = sum(lot['qty'] for lot in buy_queues[ticker])
                    if total_qty_before > 0:
                        ratio = t['Quantity'] / total_qty_before
                        for lot in buy_queues[ticker]:
                            lot['qty'] *= (1.0 + ratio)
                            lot['price'] /= (1.0 + ratio)
                    else:
                        # Fallback for anomaly cases
                        import logging
                        logging.warning(
                            f"CẢNH BÁO: Nhận cổ tức cổ phiếu cho {ticker} vào ngày {t['Date'].strftime('%d/%m/%Y')} "
                            f"nhưng hàng đợi mua trống (không nắm giữ cổ phiếu trước ngày chốt quyền). "
                            f"Điều này có thể do sai lệch dữ liệu lịch sử hoặc giao dịch."
                        )
                        buy_queues[ticker].append({
                            'qty': t['Quantity'],
                            'price': 0.0,
                            'date': t['Date'],
                            'fee': 0.0
                        })
                elif t['Action'] == 'DIVIDEND_CASH':
                    # Distribute cash dividend to active lots or completed trades
                    net_amount = t['TotalValue']
                    total_qty = sum(lot['qty'] for lot in buy_queues[ticker])
                    if total_qty > 0:
                        d = net_amount / total_qty
                        for lot in buy_queues[ticker]:
                            # Reduce cost basis of active lots by the net dividend per share
                            lot['price'] -= d
                    else:
                        # Find recently completed trades for this ticker
                        ticker_completed = [tc for tc in completed_trades if tc['ticker'] == ticker]
                        if ticker_completed:
                            last_trade = ticker_completed[-1]
                            last_trade['profit'] += net_amount
                            if 'buy_cost' in last_trade and last_trade['buy_cost'] > 0:
                                last_trade['return'] = last_trade['profit'] / last_trade['buy_cost']
                elif t['Action'] == 'SELL':
                    sell_qty = t['Quantity']
                    sell_price = t['Price']
                    sell_date = t['Date']
                    sell_fee = t['Fee']
                    sell_tax = t['Tax']
                    
                    realized_gain = 0.0
                    total_buy_cost = 0.0
                    days_held_sum = 0.0
                    matched_qty_sum = 0
                    
                    buy_queue = buy_queues[ticker]
                    while sell_qty > 1e-5 and buy_queue:
                        buy_lot = buy_queue[0]
                        matched_qty = min(sell_qty, buy_lot['qty'])
                        
                        # Calculate proportional buy cost
                        prop_buy_cost = matched_qty * buy_lot['price']
                        prop_buy_fee = buy_lot['fee'] * (matched_qty / buy_lot['qty'])
                        
                        total_buy_cost += (prop_buy_cost + prop_buy_fee)
                        
                        # Days held
                        hold_days = (sell_date - buy_lot['date']).days
                        days_held_sum += hold_days * matched_qty
                        matched_qty_sum += matched_qty
                        
                        # Deduct from buy queue and update remaining fee
                        buy_lot['fee'] -= prop_buy_fee
                        buy_lot['qty'] -= matched_qty
                        sell_qty -= matched_qty
                        if buy_lot['qty'] < 1e-5:
                            buy_queue.pop(0)
                            
                    if matched_qty_sum > 0:
                        # Proceeds of this matched portion
                        prop_sell_val = matched_qty_sum * sell_price
                        prop_sell_fee = sell_fee * (matched_qty_sum / t['Quantity'])
                        prop_sell_tax = sell_tax * (matched_qty_sum / t['Quantity'])
                        net_proceeds = prop_sell_val - prop_sell_fee - prop_sell_tax
                        
                        trade_profit = net_proceeds - total_buy_cost
                        trade_return = trade_profit / total_buy_cost if total_buy_cost > 0 else 0.0
                        avg_hold = days_held_sum / matched_qty_sum
                        
                        completed_trades.append({
                            'ticker': ticker,
                            'profit': trade_profit,
                            'return': trade_return,
                            'hold_days': avg_hold,
                            'buy_cost': total_buy_cost
                        })

            # Calculate stats from completed trades
            n_completed = len(completed_trades)
            if n_completed > 0:
                trade_returns = [tc['return'] for tc in completed_trades]
                trade_profits = [tc['profit'] for tc in completed_trades]
                
                wins = [p for p in trade_profits if p > 0]
                losses = [p for p in trade_profits if p <= 0]
                
                win_rate = len(wins) / n_completed
                
                sum_wins = float(sum(wins))
                sum_losses = float(abs(sum(losses)))
                if sum_losses > 1e-4:
                    profit_factor = sum_wins / sum_losses
                else:
                    profit_factor = float('inf') if sum_wins > 1e-4 else 0.0
                
                avg_trade_return = np.mean(trade_returns)
                best_trade = np.max(trade_returns)
                worst_trade = np.min(trade_returns)
                avg_hold_days = np.mean([tc['hold_days'] for tc in completed_trades])

        # Benchmark Metrics
        benchmark_return = 0.0
        benchmark_cagr = 0.0
        alpha = 0.0
        beta = 1.0
        outperformance = 0.0

        if benchmark_data is not None and not benchmark_data.empty:
            # Align dates
            aligned_data = pd.DataFrame(index=equity_curve.index)
            aligned_data['Strategy_Return'] = daily_returns
            
            # Map benchmark Close to aligned index
            # Drop timezone information to avoid timezone mismatches
            bench_close = benchmark_data['Close'].copy()
            bench_close.index = bench_close.index.tz_localize(None) if bench_close.index.tz is not None else bench_close.index
            strategy_index = equity_curve.index.tz_localize(None) if equity_curve.index.tz is not None else equity_curve.index
            
            bench_close_aligned = bench_close.reindex(strategy_index).ffill().bfill()
            aligned_data['Benchmark_Close'] = bench_close_aligned
            aligned_data['Benchmark_Return'] = aligned_data['Benchmark_Close'].pct_change().fillna(0.0)
            
            # Benchmark total return
            bench_start = aligned_data['Benchmark_Close'].iloc[0]
            bench_end = aligned_data['Benchmark_Close'].iloc[-1]
            benchmark_return = (bench_end - bench_start) / bench_start if bench_start > 0 else 0.0
            
            # Benchmark CAGR
            if years > 0 and bench_end > 0 and bench_start > 0:
                benchmark_cagr = (bench_end / bench_start) ** (1 / years) - 1
            else:
                benchmark_cagr = 0.0
                
            outperformance = total_return - benchmark_return
            
            # OLS Regression for Alpha and Beta using daily excess returns
            # excess returns = return - risk_free_rate / 252 (daily risk-free rate)
            daily_rf = risk_free_rate / 252.0
            excess_strat = aligned_data['Strategy_Return'] - daily_rf
            excess_bench = aligned_data['Benchmark_Return'] - daily_rf
            
            mask = ~np.isnan(excess_bench) & ~np.isnan(excess_strat)
            eb_clean = excess_bench[mask]
            es_clean = excess_strat[mask]
            
            if len(eb_clean) > 1:
                # OLS: es_clean = beta * eb_clean + alpha_daily
                beta, alpha_daily = np.polyfit(eb_clean, es_clean, 1)
                alpha = alpha_daily * 252.0 # Annualized Alpha
            else:
                beta = 1.0
                alpha = 0.0

        return {
            'duration_days': duration_days,
            'years': round(years, 2),
            'initial_cash': initial_cash,
            'final_equity': final_equity,
            'total_return': total_return,
            'cagr': cagr,
            'annualized_vol': ann_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_hold_days': round(avg_hold_days, 1),
            'benchmark_return': benchmark_return,
            'benchmark_cagr': benchmark_cagr,
            'outperformance': outperformance,
            'alpha': alpha,
            'beta': beta,
            'risk_free_rate': risk_free_rate
        }
