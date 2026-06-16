import os
import jinja2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, Union

# Use dark template for Plotly by default
pio.templates.default = "plotly_dark"

class ReportGenerator:
    """
    Generates a premium, interactive HTML report for the backtest results.
    Includes performance metrics dashboard, equity curve, drawdown, and trade signals.
    """
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_plotly_html(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, stock_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], benchmark_df: pd.DataFrame = None, benchmark_symbol: str = "VNINDEX") -> str:
        """Create the interactive Plotly figures and return their HTML snippets."""
        # 1. Chart 1: Equity Curve vs Benchmark (Normalized to 100)
        fig_equity = go.Figure()
        
        # Strategy normalized equity
        strategy_normalized = (equity_df['Equity'] / equity_df['Equity'].iloc[0]) * 100
        fig_equity.add_trace(go.Scatter(
            x=strategy_normalized.index,
            y=strategy_normalized,
            name='Chiến lược (VN-Backtest)',
            line=dict(color='#00ffcc', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 204, 0.05)'
        ))
        
        # Benchmark normalized equity
        if benchmark_df is not None and not benchmark_df.empty:
            # Drop timezone information if present
            bench_close = benchmark_df['Close'].copy()
            bench_close.index = bench_close.index.tz_localize(None) if bench_close.index.tz is not None else bench_close.index
            strategy_index = equity_df.index.tz_localize(None) if equity_df.index.tz is not None else equity_df.index
            
            bench_close_aligned = bench_close.reindex(strategy_index).ffill().bfill()
            bench_normalized = (bench_close_aligned / bench_close_aligned.iloc[0]) * 100
            fig_equity.add_trace(go.Scatter(
                x=bench_normalized.index,
                y=bench_normalized,
                name=f'Benchmark ({benchmark_symbol})',
                line=dict(color='#ff9900', width=2, dash='dash')
            ))
            
        fig_equity.update_layout(
            title='<b>TĂNG TRƯỞNG TÀI SẢN TÍCH LŨY (Chuẩn hóa về 100)</b>',
            xaxis_title='Ngày',
            yaxis_title='Giá trị tài sản (%)',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(15, 23, 42, 0.8)'),
            margin=dict(l=20, r=20, t=50, b=20),
            height=450
        )
        fig_equity.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_equity.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

        # 2. Chart 2: Drawdown Area Chart
        fig_dd = go.Figure()
        running_max = equity_df['Equity'].cummax()
        drawdown = ((equity_df['Equity'] - running_max) / running_max) * 100
        
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='Sụt giảm (Drawdown)',
            line=dict(color='#ff4d4d', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 77, 77, 0.15)'
        ))
        
        fig_dd.update_layout(
            title='<b>MỨC SỤT GIẢM TÀI SẢN (Drawdown %)</b>',
            xaxis_title='Ngày',
            yaxis_title='Sụt giảm (%)',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            height=250
        )
        fig_dd.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_dd.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-100, 5])

        # 3. Chart 3: Stock Close Price & Buy/Sell Signals (with Interactive Ticker Dropdown)
        fig_signals = go.Figure()
        
        if isinstance(stock_df, dict):
            tickers = list(stock_df.keys())
        else:
            tickers = ['PORTFOLIO']
            stock_df = {'PORTFOLIO': stock_df}
            
        ticker_trace_indices = {}
        trace_idx = 0
        
        for i, ticker in enumerate(tickers):
            ticker_trace_indices[ticker] = []
            df = stock_df[ticker]
            df_no_tz = df.copy()
            df_no_tz.index = df_no_tz.index.tz_localize(None) if df_no_tz.index.tz is not None else df_no_tz.index
            
            # Close Price Trace
            fig_signals.add_trace(go.Scatter(
                x=df_no_tz.index,
                y=df_no_tz['Close'],
                name=f'Giá đóng cửa {ticker}',
                line=dict(width=2, color='#3b82f6' if len(tickers) == 1 else None),
                opacity=0.8,
                visible=(i == 0)
            ))
            ticker_trace_indices[ticker].append(trace_idx)
            trace_idx += 1
            
            # Buy/Sell Signals Traces
            if not trades_df.empty:
                trades_df_no_tz = trades_df.copy()
                trades_df_no_tz['Date'] = pd.to_datetime(trades_df_no_tz['Date']).dt.tz_localize(None)
                
                ticker_trades = trades_df_no_tz[trades_df_no_tz['Ticker'] == ticker]
                buys = ticker_trades[ticker_trades['Action'] == 'BUY']
                sells = ticker_trades[ticker_trades['Action'] == 'SELL']
                
                # Add Buy markers
                fig_signals.add_trace(go.Scatter(
                    x=buys['Date'] if not buys.empty else [],
                    y=buys['Price'] if not buys.empty else [],
                    mode='markers',
                    name=f'MUA ({ticker})',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00ff00',
                        line=dict(color='#052e16', width=1.5)
                    ),
                    text=[f"Mua: {q} CP @ {p:,.0f}" for q, p in zip(buys['Quantity'], buys['Price'])] if not buys.empty else [],
                    hoverinfo='text+x',
                    visible=(i == 0)
                ))
                ticker_trace_indices[ticker].append(trace_idx)
                trace_idx += 1
                
                # Add Sell markers
                fig_signals.add_trace(go.Scatter(
                    x=sells['Date'] if not sells.empty else [],
                    y=sells['Price'] if not sells.empty else [],
                    mode='markers',
                    name=f'BÁN ({ticker})',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ff0000',
                        line=dict(color='#450a0a', width=1.5)
                    ),
                    text=[f"Bán: {q} CP @ {p:,.0f}" for q, p in zip(sells['Quantity'], sells['Price'])] if not sells.empty else [],
                    hoverinfo='text+x',
                    visible=(i == 0)
                ))
                ticker_trace_indices[ticker].append(trace_idx)
                trace_idx += 1
            else:
                # Add empty traces to keep structure consistent
                fig_signals.add_trace(go.Scatter(x=[], y=[], mode='markers', name=f'MUA ({ticker})', visible=(i == 0)))
                ticker_trace_indices[ticker].append(trace_idx)
                trace_idx += 1
                fig_signals.add_trace(go.Scatter(x=[], y=[], mode='markers', name=f'BÁN ({ticker})', visible=(i == 0)))
                ticker_trace_indices[ticker].append(trace_idx)
                trace_idx += 1
                
        # Generate dropdown menu buttons
        updatemenus = []
        if len(tickers) > 1:
            buttons = []
            total_traces = len(fig_signals.data)
            for i, ticker in enumerate(tickers):
                visibility = [False] * total_traces
                for idx in ticker_trace_indices[ticker]:
                    visibility[idx] = True
                
                button = dict(
                    label=ticker,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"<b>ĐIỂM GIAO DỊCH TRÊN ĐỒ THỊ GIÁ: {ticker}</b>"}
                    ]
                )
                buttons.append(button)
                
            updatemenus = [
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="rgba(15, 23, 42, 0.9)",
                    bordercolor="rgba(255, 255, 255, 0.15)",
                    font=dict(color="#f3f4f6")
                )
            ]
            
        fig_signals.update_layout(
            title=f'<b>ĐIỂM GIAO DỊCH TRÊN ĐỒ THỊ GIÁ: {tickers[0]}</b>',
            xaxis_title='Ngày',
            yaxis_title='Giá (VND)',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(15, 23, 42, 0.8)'),
            margin=dict(l=20, r=20, t=80, b=20),
            height=480,
            updatemenus=updatemenus
        )
        fig_signals.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_signals.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

        # Convert to HTML snippets (div tags)
        equity_html = pio.to_html(fig_equity, include_plotlyjs=False, full_html=False)
        dd_html = pio.to_html(fig_dd, include_plotlyjs=False, full_html=False)
        signals_html = pio.to_html(fig_signals, include_plotlyjs=False, full_html=False)
        
        return equity_html, dd_html, signals_html

    def generate_report(
        self,
        metrics: Dict[str, Any],
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        stock_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        strategy_name: str,
        benchmark_data: pd.DataFrame = None,
        filename: str = "backtest_report.html",
        benchmark_symbol: str = "VNINDEX"
    ) -> str:
        """
        Generate the beautiful, fully-styled HTML report.
        
        Returns:
            str: Path to the generated report file.
        """
        # Generate chart HTML snippets
        equity_chart, dd_chart, signals_chart = self._generate_plotly_html(
            equity_curve, trades, stock_data, benchmark_data, benchmark_symbol
        )
        
        # Prepare HTML template variables
        report_title = f"Báo cáo Backtest: {ticker} - {strategy_name}"
        
        # Convert trades to a list of dicts for rendering, limit to last 100 for size
        trades_list = []
        if not trades.empty:
            # Filter out daily margin interest logs to avoid flooding the trade history table
            real_trades = trades[trades['Action'] != 'MARGIN_INTEREST']
            trades_sorted = real_trades.sort_values('Date', ascending=False)
            for idx, r in trades_sorted.iterrows():
                note = r['Note'] if 'Note' in r.index and pd.notna(r['Note']) else ''
                trades_list.append({
                    'date': r['Date'].strftime('%d/%m/%Y'),
                    'ticker': r['Ticker'],
                    'action': r['Action'],
                    'qty': f"{r['Quantity']:,}",
                    'price': f"{r['Price']:.2f}",
                    'val': f"{r['Value']:,.0f}" if r['Value'] > 1000 else f"{r['Value']:.2f}",
                    'fee': f"{r['Fee']:.2f}",
                    'tax': f"{r['Tax']:.2f}",
                    'total': f"{r['TotalValue']:,.0f}" if r['TotalValue'] > 1000 else f"{r['TotalValue']:.2f}",
                    'note': note
                })
        
        html_content = self.HTML_TEMPLATE.render(
            title=report_title,
            ticker=ticker,
            strategy_name=strategy_name,
            metrics=metrics,
            trades=trades_list[:100],  # Show last 100 trades in table
            total_trades_count=len(trades_list),
            equity_chart=equity_chart,
            dd_chart=dd_chart,
            signals_chart=signals_chart,
            benchmark_symbol=benchmark_symbol
        )
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path

    def generate_optimization_report(
        self,
        results_df: pd.DataFrame,
        ticker: str,
        strategy_name: str,
        filename: str = "backtest_opt_report.html"
    ) -> str:
        """
        Generate a beautiful, interactive HTML report for the optimization results.
        Includes a summary of the best parameter set, a responsive leaderboard table,
        and an interactive 2D Heatmap / Bar chart using Plotly.
        """
        # 1. Identify parameter columns vs metric columns
        metric_cols = {
            'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 
            'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'
        }
        param_cols = [col for col in results_df.columns if col not in metric_cols]
        
        # Sort results by Sharpe Ratio descending
        results_sorted = results_df.sort_values(by="sharpe_ratio", ascending=False)
        best_row = results_sorted.iloc[0]
        
        # Prepare parameters description for best config
        best_params_str = ", ".join([f"{p}={best_row[p]}" for p in param_cols])
        
        # 2. Draw Plotly visualization based on number of parameter columns
        fig_opt = go.Figure()
        
        if len(param_cols) == 2:
            p1 = param_cols[0]
            p2 = param_cols[1]
            
            # Construct a pivot table for the heatmap
            pivot_df = results_df.pivot(index=p1, columns=p2, values='sharpe_ratio')
            
            fig_opt.add_trace(go.Heatmap(
                x=pivot_df.columns,
                y=pivot_df.index,
                z=pivot_df.values,
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                hovertemplate=f'{p2}: %{{x}}<br>{p1}: %{{y}}<br>Sharpe: %{{z}}:.2f<extra></extra>'
            ))
            
            fig_opt.update_layout(
                title=f'<b>BẢN ĐỒ NHIỆT SHARPE RATIO ({p1.upper()} vs {p2.upper()})</b>',
                xaxis_title=p2.upper(),
                yaxis_title=p1.upper(),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=50, b=40),
                height=450
            )
        elif len(param_cols) == 1:
            p1 = param_cols[0]
            df_sorted_p = results_df.sort_values(by=p1)
            
            fig_opt.add_trace(go.Scatter(
                x=df_sorted_p[p1],
                y=df_sorted_p['sharpe_ratio'],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='#00ffcc', width=3),
                marker=dict(size=8, color='#3b82f6')
            ))
            
            fig_opt.update_layout(
                title=f'<b>HIỆU QUẢ SHARPE RATIO THEO THAM SỐ {p1.upper()}</b>',
                xaxis_title=p1.upper(),
                yaxis_title='Sharpe Ratio',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=50, b=40),
                height=450
            )
        else:
            top_n = min(15, len(results_sorted))
            top_df = results_sorted.head(top_n)
            
            labels = []
            for _, r in top_df.iterrows():
                lbl = ", ".join([f"{p}:{r[p]}" for p in param_cols])
                labels.append(lbl)
                
            fig_opt.add_trace(go.Bar(
                x=labels,
                y=top_df['sharpe_ratio'],
                marker_color='#00ffcc',
                opacity=0.85
            ))
            
            fig_opt.update_layout(
                title='<b>BẢNG XẾP HẠNG TOP THAM SỐ (Sharpe Ratio)</b>',
                xaxis_title='Tổ hợp tham số',
                yaxis_title='Sharpe Ratio',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=50, b=100),
                height=500
            )
            
        opt_chart_html = pio.to_html(fig_opt, include_plotlyjs=False, full_html=False)
        
        # 3. Format results rows for HTML table
        leaderboard = []
        for idx, (_, r) in enumerate(results_sorted.iterrows(), 1):
            leaderboard.append({
                'rank': idx,
                'params': ", ".join([f"{p}={r[p]}" for p in param_cols]),
                'ret': f"{r['total_return']*100:.2f}%",
                'cagr': f"{r['cagr']*100:.2f}%",
                'sharpe': f"{r['sharpe_ratio']:.2f}",
                'sortino': f"{r['sortino_ratio']:.2f}",
                'drawdown': f"{r['max_drawdown']*100:.2f}%",
                'win_rate': f"{r['win_rate']*100:.1f}%",
                'profit_factor': f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞',
                'trades': int(r['total_trades'])
            })
            
        best_metrics = {
            'params': best_params_str,
            'total_return': best_row['total_return'],
            'cagr': best_row['cagr'],
            'sharpe_ratio': best_row['sharpe_ratio'],
            'sortino_ratio': best_row['sortino_ratio'],
            'max_drawdown': best_row['max_drawdown'],
            'win_rate': best_row['win_rate'],
            'profit_factor': best_row['profit_factor'] if best_row['profit_factor'] != float('inf') else 999.0,
            'total_trades': int(best_row['total_trades'])
        }
        
        report_title = f"Báo cáo Tối ưu hóa: {ticker} - {strategy_name}"
        
        html_content = self.HTML_OPTIMIZATION_TEMPLATE.render(
            title=report_title,
            ticker=ticker,
            strategy_name=strategy_name,
            best=best_metrics,
            leaderboard=leaderboard,
            chart_html=opt_chart_html
        )
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path

    # Jinja2 HTML template with premium design styling (glassmorphism dark mode)
    HTML_TEMPLATE = jinja2.Template("""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <!-- Include Google Font Inter and Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Include Plotly JS CDN -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 25, 40, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #00ffcc;
            --primary-hover: #00cc99;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1.5rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header Style */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .header-title h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00ffcc 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .header-title p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        .badge {
            background: rgba(0, 255, 204, 0.15);
            color: var(--primary);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(0, 255, 204, 0.3);
            text-transform: uppercase;
        }

        /* Glassmorphism Card Layout */
        .grid-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 255, 204, 0.25);
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .metric-subtext {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Colored metrics */
        .val-positive {
            color: var(--success);
        }

        .val-negative {
            color: var(--danger);
        }

        /* Charts Layout */
        .grid-charts {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }

        @media (min-width: 1024px) {
            .grid-charts {
                grid-template-columns: 2fr 1fr;
            }
            .full-width-chart {
                grid-column: span 2;
            }
        }

        .chart-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        /* Table Style */
        .table-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 2rem;
            overflow-x: auto;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        .table-container h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            margin-bottom: 1.2rem;
            color: var(--text-main);
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        th {
            color: var(--text-muted);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }

        td {
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            color: #d1d5db;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-main);
        }

        .badge-buy {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-sell {
            background: rgba(239, 68, 68, 0.15);
            color: var(--danger);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-dividend {
            background: rgba(139, 92, 246, 0.15);
            color: #a78bfa;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="header-title">
                <h1>{{ ticker }} - {{ strategy_name }}</h1>
                <p>Hệ thống VN-Backtest • Khung thời gian: {{ metrics.years }} năm ({{ metrics.duration_days }} ngày)</p>
            </div>
            <div>
                <span class="badge">Việt Nam Stock Market</span>
            </div>
        </header>

        <!-- Survival Bias Disclaimer -->
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 0.75rem 1.25rem; margin-bottom: 2rem; display: flex; align-items: center; gap: 0.75rem; font-size: 0.85rem; color: #fef08a;">
            <svg style="width: 16px; height: 16px; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"/></svg>
            <div>
                <strong>Cảnh báo dữ liệu (Survival Bias):</strong> Kết quả backtest dựa trên các mã cổ phiếu hiện đang hoạt động trực tuyến. Để kiểm thử các cổ phiếu đã hủy niêm yết (như ROS, FLC), vui lòng đặt tệp dữ liệu vào thư mục <code>local_data/</code>.
            </div>
        </div>

        <!-- DASHBOARD METRICS -->
        <div class="grid-metrics">
            <!-- Total Return -->
            <div class="card">
                <div class="metric-label">Tổng Lợi Nhuận</div>
                <div class="metric-value {% if metrics.total_return >= 0 %}val-positive{% else %}val-negative{% endif %}">
                    {% if metrics.total_return >= 0 %}+{% endif %}{{ (metrics.total_return * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">Benchmark {{ benchmark_symbol | default('VNINDEX') }}: {{ (metrics.benchmark_return * 100) | round(2) }}%</div>
            </div>
            
            <!-- CAGR -->
            <div class="card">
                <div class="metric-label">Lợi Nhuận Gộp Hàng Năm (CAGR)</div>
                <div class="metric-value {% if metrics.cagr >= 0 %}val-positive{% else %}val-negative{% endif %}">
                    {% if metrics.cagr >= 0 %}+{% endif %}{{ (metrics.cagr * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">Đã bao gồm thuế và phí giao dịch</div>
            </div>

            <!-- Max Drawdown -->
            <div class="card">
                <div class="metric-label">Sụt Giảm Lớn Nhất (MDD)</div>
                <div class="metric-value val-negative">
                    {{ (metrics.max_drawdown * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">Thời gian phục hồi dài nhất: {{ metrics.max_drawdown_duration }} phiên</div>
            </div>

            <!-- Sharpe & Sortino -->
            <div class="card">
                <div class="metric-label">Sharpe / Sortino Ratio</div>
                <div class="metric-value">
                    {{ metrics.sharpe_ratio | round(2) }} / {{ metrics.sortino_ratio | round(2) }}
                </div>
                <div class="metric-subtext">Tính trên lãi suất phi rủi ro: {{ (metrics.risk_free_rate or 0.04) * 100 }}%</div>
            </div>

            <!-- Win Rate & Profit Factor -->
            <div class="card">
                <div class="metric-label">Tỷ Lệ Thắng / Profit Factor</div>
                <div class="metric-value">
                    {{ (metrics.win_rate * 100) | round(1) }}% / {{ metrics.profit_factor | round(2) }}
                </div>
                <div class="metric-subtext">Tổng số lệnh khớp: {{ metrics.total_trades }} (Giữ TB: {{ metrics.avg_hold_days }} ngày)</div>
            </div>

            <!-- Alpha & Beta -->
            <div class="card">
                <div class="metric-label">Hệ Số Alpha & Beta</div>
                <div class="metric-value">
                    {% if metrics.alpha >= 0 %}+{% endif %}{{ (metrics.alpha * 100) | round(2) }}% / {{ metrics.beta | round(2) }}
                </div>
                <div class="metric-subtext">Vượt trội so với thị trường: {{ (metrics.outperformance * 100) | round(2) }}%</div>
            </div>
        </div>

        <!-- CHARTS SECTION -->
        <div class="grid-charts">
            <!-- Equity curve vs benchmark -->
            <div class="chart-container">
                {{ equity_chart }}
            </div>
            
            <!-- Drawdown area -->
            <div class="chart-container">
                {{ dd_chart }}
            </div>

            <!-- Full width signals chart -->
            <div class="chart-container full-width-chart">
                {{ signals_chart }}
            </div>
        </div>

        <!-- LATEST TRADES -->
        <div class="table-container">
            <h3>Nhật ký giao dịch gần đây (Tối đa 100 lệnh mới nhất / Tổng {{ total_trades_count }} lệnh)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Ngày khớp</th>
                        <th>Mã CP</th>
                        <th>Loại lệnh</th>
                        <th>Số lượng</th>
                        <th>Giá khớp</th>
                        <th>Giá trị khớp</th>
                        <th>Phí GD</th>
                        <th>Thuế bán</th>
                        <th>Thực nhận / Chi</th>
                        <th>Ghi chú</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr>
                        <td>{{ trade.date }}</td>
                        <td>{{ trade.ticker }}</td>
                        <td>
                            <span class="{% if trade.action == 'BUY' %}badge-buy{% elif trade.action == 'SELL' %}badge-sell{% else %}badge-dividend{% endif %}">
                                {{ trade.action }}
                            </span>
                        </td>
                        <td>{{ trade.qty }}</td>
                        <td>{{ trade.price }}</td>
                        <td>{{ trade.val }}</td>
                        <td>{{ trade.fee }}</td>
                        <td>{{ trade.tax }}</td>
                        <td class="{% if trade.action in ['SELL', 'DIVIDEND_CASH', 'DIVIDEND_STOCK'] %}val-positive{% endif %}">
                            {{ trade.total }}
                        </td>
                        <td>{{ trade.note }}</td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="9" style="text-align: center; color: var(--text-muted);">Không có giao dịch nào được thực hiện.</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """)

    # Jinja2 HTML template with premium design styling for optimization results
    HTML_OPTIMIZATION_TEMPLATE = jinja2.Template("""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <!-- Include Google Font Inter and Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Include Plotly JS CDN -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 25, 40, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #00ffcc;
            --primary-hover: #00cc99;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1.5rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header Style */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .header-title h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00ffcc 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .header-title p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        .badge {
            background: rgba(0, 255, 204, 0.15);
            color: var(--primary);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(0, 255, 204, 0.3);
            text-transform: uppercase;
        }

        /* Glassmorphism Card Layout */
        .grid-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 255, 204, 0.25);
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .metric-subtext {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Colored metrics */
        .val-positive {
            color: var(--success);
        }

        .val-negative {
            color: var(--danger);
        }

        .best-params-badge {
            background: rgba(0, 255, 204, 0.1);
            color: var(--primary);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            border: 1px solid rgba(0, 255, 204, 0.2);
            font-family: monospace;
            font-size: 1.1rem;
            display: inline-block;
            margin-top: 0.3rem;
        }

        /* Charts Layout */
        .chart-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 2.5rem;
        }

        /* Table Style */
        .table-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            overflow-x: auto;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        .table-container h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            margin-bottom: 1.2rem;
            color: var(--text-main);
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        th {
            color: var(--text-muted);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }

        td {
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            color: #d1d5db;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-main);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="header-title">
                <h1>{{ ticker }} - Tối Ưu Hóa Tham Số</h1>
                <p>Chiến lược: {{ strategy_name }} • Grid Search Parameter Optimization</p>
            </div>
            <div>
                <span class="badge">Optimization Report</span>
            </div>
        </header>

        <!-- DASHBOARD METRICS -->
        <div class="grid-metrics">
            <div class="card" style="grid-column: span 2;">
                <div class="metric-label">Tổ hợp tham số tốt nhất (Sharpe cao nhất)</div>
                <div class="best-params-badge">{{ best.params }}</div>
                <div class="metric-subtext" style="margin-top: 0.5rem;">Cơ sở đánh giá xếp hạng: Sharpe Ratio cao nhất.</div>
            </div>
            
            <div class="card">
                <div class="metric-label">Tổng Lợi Nhuận (Best)</div>
                <div class="metric-value val-positive">
                    +{{ (best.total_return * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">CAGR: {{ (best.cagr * 100) | round(2) }}%</div>
            </div>

            <div class="card">
                <div class="metric-label">Sharpe / Sortino (Best)</div>
                <div class="metric-value">
                    {{ best.sharpe_ratio | round(2) }} / {{ best.sortino_ratio | round(2) }}
                </div>
                <div class="metric-subtext">Hệ số hiệu quả điều chỉnh rủi ro</div>
            </div>

            <div class="card">
                <div class="metric-label">Sụt giảm lớn nhất (Best)</div>
                <div class="metric-value val-negative">
                    {{ (best.max_drawdown * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">Số lượng khớp: {{ best.total_trades }} giao dịch</div>
            </div>
        </div>

        <!-- PLOTLY HEATMAP/CHART -->
        <div class="chart-container">
            {{ chart_html }}
        </div>

        <!-- LEADERBOARD TABLE -->
        <div class="table-container">
            <h3>Bảng xếp hạng hiệu năng kịch bản</h3>
            <table>
                <thead>
                    <tr>
                        <th>Hạng</th>
                        <th>Tổ Hợp Tham Số</th>
                        <th>Tổng Lợi Nhuận</th>
                        <th>CAGR (Năm)</th>
                        <th>Hệ Số Sharpe</th>
                        <th>Hệ Số Sortino</th>
                        <th>Sụt Giảm Lớn Nhất</th>
                        <th>Tỷ Lệ Thắng</th>
                        <th>Profit Factor</th>
                        <th>Số Lệnh</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in leaderboard %}
                    <tr style="{% if row.rank == 1 %}background: rgba(0, 255, 204, 0.05); font-weight: 600;{% endif %}">
                        <td>#{{ row.rank }}</td>
                        <td style="font-family: monospace; color: var(--primary);">{{ row.params }}</td>
                        <td class="val-positive">{{ row.ret }}</td>
                        <td>{{ row.cagr }}</td>
                        <td style="color: #ffffff; font-weight: 500;">{{ row.sharpe }}</td>
                        <td>{{ row.sortino }}</td>
                        <td class="val-negative">{{ row.drawdown }}</td>
                        <td>{{ row.win_rate }}</td>
                        <td>{{ row.profit_factor }}</td>
                        <td>{{ row.trades }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """)
