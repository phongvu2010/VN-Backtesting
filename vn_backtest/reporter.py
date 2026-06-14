import os
import jinja2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any

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

    def _generate_plotly_html(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> str:
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
                name='Benchmark (VN-Index)',
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

        # 3. Chart 3: Stock Close Price & Buy/Sell Signals
        fig_signals = go.Figure()
        
        # Stock close price
        # Drop timezone from stock_df index if needed
        stock_df_no_tz = stock_df.copy()
        stock_df_no_tz.index = stock_df_no_tz.index.tz_localize(None) if stock_df_no_tz.index.tz is not None else stock_df_no_tz.index
        
        fig_signals.add_trace(go.Scatter(
            x=stock_df_no_tz.index,
            y=stock_df_no_tz['Close'],
            name='Giá đóng cửa',
            line=dict(color='#3b82f6', width=2),
            opacity=0.75
        ))
        
        # Filter buy/sell trades
        if not trades_df.empty:
            trades_df_no_tz = trades_df.copy()
            trades_df_no_tz['Date'] = pd.to_datetime(trades_df_no_tz['Date']).dt.tz_localize(None)
            
            buys = trades_df_no_tz[trades_df_no_tz['Action'] == 'BUY']
            sells = trades_df_no_tz[trades_df_no_tz['Action'] == 'SELL']
            
            # Add Buy Markers
            fig_signals.add_trace(go.Scatter(
                x=buys['Date'],
                y=buys['Price'],
                mode='markers',
                name='Lệnh Mua (BUY)',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#00ff00',
                    line=dict(color='#052e16', width=1.5)
                ),
                text=[f"Mua: {q} CP @ {p:.2f}" for q, p in zip(buys['Quantity'], buys['Price'])],
                hoverinfo='text+x'
            ))
            
            # Add Sell Markers
            fig_signals.add_trace(go.Scatter(
                x=sells['Date'],
                y=sells['Price'],
                mode='markers',
                name='Lệnh Bán (SELL)',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#ff0000',
                    line=dict(color='#450a0a', width=1.5)
                ),
                text=[f"Bán: {q} CP @ {p:.2f}" for q, p in zip(sells['Quantity'], sells['Price'])],
                hoverinfo='text+x'
            ))

        fig_signals.update_layout(
            title='<b>ĐIỂM GIAO DỊCH TRÊN ĐỒ THỊ GIÁ</b>',
            xaxis_title='Ngày',
            yaxis_title='Giá (nghìn VND)',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(15, 23, 42, 0.8)'),
            margin=dict(l=20, r=20, t=50, b=20),
            height=450
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
        stock_data: pd.DataFrame,
        ticker: str,
        strategy_name: str,
        benchmark_data: pd.DataFrame = None,
        filename: str = "backtest_report.html"
    ) -> str:
        """
        Generate the beautiful, fully-styled HTML report.
        
        Returns:
            str: Path to the generated report file.
        """
        # Generate chart HTML snippets
        equity_chart, dd_chart, signals_chart = self._generate_plotly_html(
            equity_curve, trades, stock_data, benchmark_data
        )
        
        # Prepare HTML template variables
        report_title = f"Báo cáo Backtest: {ticker} - {strategy_name}"
        
        # Convert trades to a list of dicts for rendering, limit to last 100 for size
        trades_list = []
        if not trades.empty:
            trades_sorted = trades.sort_values('Date', ascending=False)
            for idx, r in trades_sorted.iterrows():
                trades_list.append({
                    'date': r['Date'].strftime('%d/%m/%Y'),
                    'action': r['Action'],
                    'qty': f"{r['Quantity']:,}",
                    'price': f"{r['Price']:.2f}",
                    'val': f"{r['Value']:,.0f}" if r['Value'] > 1000 else f"{r['Value']:.2f}",
                    'fee': f"{r['Fee']:.2f}",
                    'tax': f"{r['Tax']:.2f}",
                    'total': f"{r['TotalValue']:,.0f}" if r['TotalValue'] > 1000 else f"{r['TotalValue']:.2f}"
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
            signals_chart=signals_chart
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

        <!-- DASHBOARD METRICS -->
        <div class="grid-metrics">
            <!-- Total Return -->
            <div class="card">
                <div class="metric-label">Tổng Lợi Nhuận</div>
                <div class="metric-value {% if metrics.total_return >= 0 %}val-positive{% else %}val-negative{% endif %}">
                    {% if metrics.total_return >= 0 %}+{% endif %}{{ (metrics.total_return * 100) | round(2) }}%
                </div>
                <div class="metric-subtext">Benchmark VN-Index: {{ (metrics.benchmark_return * 100) | round(2) }}%</div>
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
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr>
                        <td>{{ trade.date }}</td>
                        <td>{{ ticker }}</td>
                        <td>
                            <span class="{% if trade.action == 'BUY' %}badge-buy{% else %}badge-sell{% endif %}">
                                {{ trade.action }}
                            </span>
                        </td>
                        <td>{{ trade.qty }}</td>
                        <td>{{ trade.price }}</td>
                        <td>{{ trade.val }}</td>
                        <td>{{ trade.fee }}</td>
                        <td>{{ trade.tax }}</td>
                        <td class="{% if trade.action == 'SELL' %}val-positive{% endif %}">
                            {{ trade.total }}
                        </td>
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
