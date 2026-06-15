import os
import argparse
from datetime import datetime
import pandas as pd
from vn_backtest.data import VNStockDataLoader
from vn_backtest.engine import BacktestEngine
from vn_backtest.strategies.ma_cross import MACrossover
from vn_backtest.analysis import PerformanceAnalyzer
from vn_backtest.reporter import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Chương trình Backtesting Chứng khoán Việt Nam (VN-Backtest)")
    parser.add_argument("--ticker", type=str, default="FPT", help="Mã cổ phiếu cần backtest (mặc định: FPT)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Ngày bắt đầu (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-06-01", help="Ngày kết thúc (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000_000.0, help="Số vốn ban đầu (mặc định: 100M VND)")
    parser.add_argument("--exchange", type=str.lower, default="hose", choices=["hose", "hnx", "upcom"], help="Sàn giao dịch để áp biên độ trần/sàn (mặc định: hose)")
    parser.add_argument("--t_settle", type=int, default=2, help="Chu kỳ thanh toán cổ phiếu (mặc định: T+2)")
    parser.add_argument("--lot_size", type=int, default=100, help="Lô giao dịch tối thiểu (mặc định: 100)")
    parser.add_argument("--fee", type=float, default=0.0015, help="Phí giao dịch mua/bán (mặc định: 0.15%%)")
    parser.add_argument("--tax", type=float, default=0.001, help="Thuế bán chứng khoán (mặc định: 0.1%%)")
    parser.add_argument("--no_cache", action="store_true", help="Không sử dụng cache dữ liệu, tải mới hoàn toàn")
    parser.add_argument("--no_dynamic", action="store_true", help="Vô hiệu hóa quy tắc lịch sử động (chạy tĩnh)")
    parser.add_argument("--allow_odd_lot", action="store_true", help="Cho phép giao dịch lô lẻ (1 cổ phiếu)")
    parser.add_argument("--max_vol_ratio", type=float, default=None, help="Tỷ lệ khối lượng giao dịch tối đa so với Volume ngày (ví dụ: 0.1)")
    parser.add_argument("--exclude_auto_close", action="store_true", help="Loại bỏ các giao dịch tất toán cuối kỳ khỏi thống kê giao dịch")
    parser.add_argument("--adjust_corp_actions", action="store_true", help="Kích hoạt mô phỏng cổ tức/chia tách (chỉ dùng nếu giá đầu vào chưa điều chỉnh)")
    parser.add_argument("--margin_ratio", type=float, default=1.0, help="Tỷ lệ ký quỹ (ví dụ: 0.5 là đòn bẩy 2x, mặc định: 1.0 - không margin)")
    parser.add_argument("--margin_interest", type=float, default=0.13, help="Lãi suất margin năm (mặc định: 13%%)")
    parser.add_argument("--margin_maintenance", type=float, default=0.35, help="Tỷ lệ ký quỹ duy trì giải chấp (mặc định: 35%%)")
    parser.add_argument("--rf_rate", type=float, default=0.04, help="Lãi suất phi rủi ro năm (mặc định: 4%%)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"KHỞI CHẠY HỆ THỐNG VN-BACKTEST CHO MÃ: {args.ticker.upper()}")
    print(f"Thời gian: {args.start} -> {args.end}")
    print(f"Vốn ban đầu: {args.cash:,.0f} VND")
    config_str = f"Cấu hình: Sàn {args.exchange.upper()} (Biên độ trần/sàn)"
    if not args.no_dynamic:
        config_str += ", Quy tắc lịch sử động (T+2.5/T+3 & Lô 1/10/100)"
    else:
        config_str += f", T+{args.t_settle}, Lô {args.lot_size} (Cấu hình tĩnh)"
    if args.allow_odd_lot:
        config_str += ", Cho phép lô lẻ (lô 1)"
    if args.max_vol_ratio is not None:
        config_str += f", Giới hạn thanh khoản {args.max_vol_ratio*100:.1f}% Volume"
    if args.adjust_corp_actions:
        config_str += ", Mô phỏng cổ tức/chia tách"
    if args.margin_ratio < 1.0:
        config_str += f", Vay Margin (Ký quỹ {args.margin_ratio*100:.0f}%, Lãi {args.margin_interest*100:.1f}%, Call {args.margin_maintenance*100:.0f}%)"
    print(config_str)
    print(f"Chi phí: Phí GD {args.fee*100:.2f}%, Thuế bán {args.tax*100:.2f}%")
    print("=" * 60)
    
    # 1. Load data
    loader = VNStockDataLoader()
    
    # Process multiple tickers
    tickers = [t.strip().upper() for t in args.ticker.split(',')]
    
    # Fetch exchange dynamically from vnstock if possible
    _exchange_map = {}
    print("-> Đang tải danh sách sàn giao dịch từ vnstock...")
    for source in ['VCI', 'KBS', 'MSN']:
        try:
            from vnstock import Listing
            l = Listing(source=source)
            df_symbols = l.symbols_by_exchange('HOSE')
            if df_symbols is not None and not df_symbols.empty and 'symbol' in df_symbols.columns and 'exchange' in df_symbols.columns:
                df_symbols = df_symbols.dropna(subset=['symbol', 'exchange'])
                for _, row in df_symbols.iterrows():
                    symbol = str(row['symbol']).upper()
                    exch = str(row['exchange']).lower()
                    if exch == 'comup':
                        exch = 'upcom'
                    elif exch == 'xhnf':
                        exch = 'hnx'
                    _exchange_map[symbol] = exch
                print(f"   Đã tải thành công {len(_exchange_map)} mã từ vnstock để cấu hình sàn (nguồn: {source}).")
                break
        except Exception as e:
            print(f"   (Cảnh báo: Lấy sàn tự động từ nguồn {source} không thành công: {e})")
    else:
        print("   CẢNH BÁO: Tất cả các nguồn dữ liệu vnstock đều không tải được danh sách sàn. Sẽ dùng danh sách mặc định.")

    stock_data = {}
    exchanges = {}
    
    print("\n[1/4] Đang tải dữ liệu các cổ phiếu...")
    for ticker in tickers:
        print(f"-> Đang tải dữ liệu cho mã: {ticker}...")
        try:
            df = loader.fetch_data(
                symbol=ticker, 
                start_date=args.start, 
                end_date=args.end, 
                is_index=False, 
                use_cache=not args.no_cache
            )
            stock_data[ticker] = df
            
            # Exchange guess helper
            def guess_exchange(t: str, default_exchange: str) -> str:
                t_upper = t.upper()
                if t_upper in _exchange_map:
                    exch = _exchange_map[t_upper]
                    if exch in ['hose', 'hnx', 'upcom']:
                        return exch
                    elif exch == 'comup':
                        return 'upcom'
                
                upcom_tickers = {
                    'ACV', 'VEA', 'BSR', 'VGI', 'MVN', 'MCH', 'QNS', 'FOX', 'LTG', 'MML', 'VTP', 
                    'OIL', 'DVN', 'SGB', 'KLB', 'BAB', 'BVB', 'ABB', 'NAB', 'VBB', 'C4G', 'BDT'
                }
                hnx_tickers = {
                    'IDC', 'PVS', 'SHS', 'MBS', 'CEO', 'HUT', 'TNG', 'DTD', 'BVS', 'LAS', 'TAR', 
                    'PVI', 'PVC', 'PVB', 'VCS', 'PGS', 'PLC', 'CAP', 'NSH', 'L14', 'VIG', 'APS'
                }
                if t_upper in upcom_tickers:
                    return "upcom"
                elif t_upper in hnx_tickers:
                    return "hnx"
                return default_exchange.lower()

            exchanges[ticker] = guess_exchange(ticker, args.exchange)
            print(f"   Đã tải {len(df)} phiên giao dịch cổ phiếu {ticker} (Sàn: {exchanges[ticker].upper()}).")
        except Exception as e:
            print(f"LỖI: Không thể tải dữ liệu cho mã {ticker}: {e}")
            return

    print("\n[2/4] Đang tải dữ liệu benchmark VN-Index...")
    benchmark_data = None
    try:
        benchmark_data = loader.fetch_data(
            symbol="VNINDEX", 
            start_date=args.start, 
            end_date=args.end, 
            is_index=True, 
            use_cache=not args.no_cache
        )
        print(f"-> Đã tải {len(benchmark_data)} phiên giao dịch VN-Index.")
        # Check date range overlap
        if not benchmark_data.empty:
            bench_start = benchmark_data.index[0]
            bench_end = benchmark_data.index[-1]
            req_start = pd.to_datetime(args.start)
            req_end = pd.to_datetime(args.end)
            if bench_start > req_start or bench_end < req_end:
                print(f"   CẢNH BÁO: Giai đoạn dữ liệu VN-Index ({bench_start.strftime('%Y-%m-%d')} -> {bench_end.strftime('%Y-%m-%d')}) "
                      f"không bao phủ hoàn toàn khoảng thời gian yêu cầu ({args.start} -> {args.end}). "
                      f"Các chỉ số Alpha/Beta so sánh có thể bị ảnh hưởng.")
    except Exception as e:
        print(f"CẢNH BÁO: Không thể tải VN-Index làm benchmark ({e}). Sẽ không so sánh với VN-Index.")

    # 2. Setup Backtest Engine
    print("\n[3/4] Đang khởi chạy mô phỏng backtest...")
    engine = BacktestEngine(
        data=stock_data,
        strategy_class=MACrossover,
        initial_cash=args.cash,
        buy_fee=args.fee,
        sell_fee=args.fee,
        sell_tax=args.tax,
        settlement_days=args.t_settle,
        lot_size=args.lot_size,
        exchange=exchanges,
        execution_at="open",
        restrict_ceiling_buy=True,
        restrict_floor_sell=True,
        slippage=0.0,
        dynamic_rules=not args.no_dynamic,
        advance_interest_rate=0.12,
        auto_close_at_end=True,
        allow_odd_lot=args.allow_odd_lot,
        max_volume_ratio=args.max_vol_ratio,
        adjust_corporate_actions=args.adjust_corp_actions,
        margin_ratio=args.margin_ratio,
        margin_interest_rate=args.margin_interest,
        margin_maintenance_ratio=args.margin_maintenance
    )
    # Inject tickers information
    engine.ticker = ",".join(tickers)
    
    results = engine.run()
    
    # 3. Analyze performance
    print("[4/4] Đang tính toán các chỉ số hiệu suất...")
    metrics = PerformanceAnalyzer.calculate_metrics(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        benchmark_data=benchmark_data,
        initial_cash=results['initial_cash'],
        risk_free_rate=args.rf_rate,
        include_auto_close=not args.exclude_auto_close
    )
    
    # 4. Print Summary to console
    print("\n" + "=" * 60)
    print(" KẾT QUẢ BACKTEST CHI TIẾT (VN-BACKTEST)")
    print("=" * 60)
    print(f"Vốn khởi đầu      : {metrics['initial_cash']:,.0f} VND")
    print(f"Giá trị tài sản cuối: {metrics['final_equity']:,.0f} VND")
    print(f"Tổng lợi nhuận    : {metrics['total_return']*100:.2f}%")
    print(f"CAGR (Lợi nhuận năm): {metrics['cagr']*100:.2f}%")
    print(f"Mức sụt giảm lớn nhất (MDD): {metrics['max_drawdown']*100:.2f}%")
    print(f"Thời gian phục hồi dài nhất: {metrics['max_drawdown_duration']} phiên")
    print(f"Tỷ lệ biến động năm : {metrics['annualized_vol']*100:.2f}%")
    print(f"Hệ số Sharpe       : {metrics['sharpe_ratio']:.2f}")
    print(f"Hệ số Sortino      : {metrics['sortino_ratio']:.2f}")
    print("-" * 60)
    print(f"Tổng số giao dịch khớp : {metrics['total_trades']}")
    print(f"Tỷ lệ thắng (Win Rate)  : {metrics['win_rate']*100:.1f}%")
    print(f"Hệ số lợi nhuận (Profit Factor): {metrics['profit_factor']:.2f}")
    if 'avg_trade_return' in metrics:
        print(f"Hiệu suất TB mỗi lệnh  : {metrics['avg_trade_return']*100:.2f}%")
        print(f"Lệnh thắng lớn nhất    : {metrics['best_trade']*100:.2f}%")
        print(f"Lệnh thua lớn nhất     : {metrics['worst_trade']*100:.2f}%")
        print(f"Thời gian giữ TB       : {metrics['avg_hold_days']} ngày")
    
    if benchmark_data is not None:
        print("-" * 60)
        print(f"Lợi nhuận VN-Index     : {metrics['benchmark_return']*100:.2f}%")
        print(f"Lợi nhuận năm VN-Index : {metrics['benchmark_cagr']*100:.2f}%")
        print(f"Outperformance         : {metrics['outperformance']*100:.2f}%")
        print(f"Hệ số Beta             : {metrics['beta']:.2f}")
        print(f"Hệ số Alpha (Hằng số)  : {metrics['alpha']*100:.2f}%")
    print("=" * 60)
    
    # 5. Generate beautiful HTML Report
    reporter = ReportGenerator()
    report_filename = f"report_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    report_path = reporter.generate_report(
        metrics=metrics,
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        stock_data=stock_data,
        ticker=",".join(tickers),
        strategy_name="MA Crossover (10/20)",
        benchmark_data=benchmark_data,
        filename=report_filename
    )
    
    # Get absolute path for output message
    abs_report_path = os.path.abspath(report_path)
    print(f"\n[OK] Đã xuất báo cáo đồ thị tương tác HTML thành công!")
    print(f"-> Đường dẫn báo cáo: file://{abs_report_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
