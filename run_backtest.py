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
    parser.add_argument("--force_adjusted", type=str.lower, default="auto", choices=["auto", "true", "false"], help="Ép buộc trạng thái điều chỉnh giá (auto: tự động phát hiện, true: đã điều chỉnh, false: chưa điều chỉnh)")
    parser.add_argument("--stop_loss", type=float, default=None, help="Tỷ lệ cắt lỗ Stop Loss (ví dụ: 0.07 cho 7%%)")
    parser.add_argument("--trailing_stop", type=float, default=None, help="Tỷ lệ chặn lãi Trailing Stop (ví dụ: 0.1 cho 10%%)")
    parser.add_argument("--optimize", action="store_true", help="Chạy tối ưu hóa tham số chiến lược (Grid Search)")
    parser.add_argument("--rebalance_interval", type=int, default=None, help="Chu kỳ cơ cấu tỷ trọng danh mục theo số phiên (ví dụ: 20)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Số tiến trình chạy song song khi tối ưu hóa (mặc định: -1 - sử dụng hết CPU)")
    parser.add_argument("--execution_at", type=str.lower, default="open", choices=["open", "close", "average", "vwap", "hl2", "typical"], help="Mô hình giá khớp lệnh (open, close, average/vwap, hl2, typical)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Tỷ lệ trượt giá cố định (ví dụ: 0.001 cho 0.1%%)")
    parser.add_argument("--market_impact", type=float, default=0.0, help="Hệ số tác động thị trường gây trượt giá động (ví dụ: 0.1)")
    parser.add_argument("--rights_listing_delay", type=int, default=90, help="Số ngày chờ giải tỏa cổ tức cổ phiếu/quyền mua (mặc định: 90 ngày)")
    parser.add_argument("--report_name", type=str, default=None, help="Tên file báo cáo HTML đầu ra (ví dụ: report_123.html)")
    
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
    config_str += f", Khớp lệnh: {args.execution_at.upper()}"
    if args.slippage > 0:
        config_str += f", Trượt giá: {args.slippage*100:.2f}%"
    if args.market_impact > 0:
        config_str += f", Hệ số tác động TT: {args.market_impact:.2f}"
    if args.allow_odd_lot:
        config_str += ", Cho phép lô lẻ (lô 1)"
    if args.max_vol_ratio is not None:
        config_str += f", Giới hạn thanh khoản {args.max_vol_ratio*100:.1f}% Volume"
    if args.adjust_corp_actions:
        config_str += f", Mô phỏng cổ tức/chia tách (Force Adjusted: {args.force_adjusted.upper()})"
    if args.margin_ratio < 1.0:
        config_str += f", Vay Margin (Ký quỹ {args.margin_ratio*100:.0f}%, Lãi {args.margin_interest*100:.1f}%, Call {args.margin_maintenance*100:.0f}%)"
    if args.rebalance_interval is not None:
        config_str += f", Cơ cấu danh mục ({args.rebalance_interval} phiên/lần)"
    print(config_str)
    print(f"Chi phí: Phí GD {args.fee*100:.2f}%, Thuế bán {args.tax*100:.2f}%")
    print("=" * 60)
    
    # 1. Load data
    loader = VNStockDataLoader()
    
    # Process multiple tickers
    tickers = [t.strip().upper() for t in args.ticker.split(',')]
    
    # Fetch exchange map using loader with caching
    print("-> Đang tải danh sách sàn giao dịch...")
    _exchange_map = loader.fetch_exchange_map(use_cache=not args.no_cache)
    print(f"   Đã tải thành công {len(_exchange_map)} mã sàn giao dịch.")

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

    # Prepare strategy parameters (Stop Loss, Trailing Stop, Rebalance Interval)
    strat_params = {}
    if args.stop_loss is not None:
        strat_params['stop_loss'] = args.stop_loss
    if args.trailing_stop is not None:
        strat_params['trailing_stop'] = args.trailing_stop
    if args.rebalance_interval is not None:
        strat_params['rebalance_interval'] = args.rebalance_interval

    # 2. Check if Parameter Optimization mode is requested
    if args.optimize:
        from vn_backtest.optimizer import ParameterOptimizer
        
        # Grid Search for fast_period and slow_period on MACrossover
        param_grid = {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50]
        }
        
        engine_kwargs = {
            'buy_fee': args.fee,
            'sell_fee': args.fee,
            'sell_tax': args.tax,
            'settlement_days': args.t_settle,
            'lot_size': args.lot_size,
            'execution_at': args.execution_at,
            'restrict_ceiling_buy': True,
            'restrict_floor_sell': True,
            'slippage': args.slippage,
            'market_impact_coef': args.market_impact,
            'dynamic_rules': not args.no_dynamic,
            'advance_interest_rate': 0.12,
            'auto_close_at_end': True,
            'allow_odd_lot': args.allow_odd_lot,
            'max_volume_ratio': args.max_vol_ratio,
            'adjust_corporate_actions': args.adjust_corp_actions,
            'force_adjusted': None if args.force_adjusted == "auto" else (args.force_adjusted == "true"),
            'margin_ratio': args.margin_ratio,
            'margin_interest_rate': args.margin_interest,
            'margin_maintenance_ratio': args.margin_maintenance,
            'strategy_params': strat_params,
            'rights_listing_delay': args.rights_listing_delay
        }
        
        optimizer = ParameterOptimizer(
            data=stock_data,
            strategy_class=MACrossover,
            param_grid=param_grid,
            initial_cash=args.cash,
            exchange=exchanges,
            benchmark_data=benchmark_data,
            risk_free_rate=args.rf_rate,
            engine_kwargs=engine_kwargs,
            n_jobs=args.n_jobs
        )
        
        results_df = optimizer.run_optimization(sort_by="sharpe_ratio", ascending=False)
        
        # Generate beautiful HTML Optimization Report
        reporter = ReportGenerator()
        if args.report_name:
            report_filename = args.report_name
        else:
            report_filename = f"report_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = reporter.generate_optimization_report(
            results_df=results_df,
            ticker=",".join(tickers),
            strategy_name="MA Crossover (SMA Fast vs Slow)",
            filename=report_filename
        )
        
        abs_report_path = os.path.abspath(report_path)
        print(f"\n[OK] Đã xuất báo cáo tối ưu hóa HTML thành công!")
        print(f"-> Báo cáo tối ưu hóa: file://{abs_report_path}")
        print("=" * 60 + "\n")
        return

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
        execution_at=args.execution_at,
        restrict_ceiling_buy=True,
        restrict_floor_sell=True,
        slippage=args.slippage,
        market_impact_coef=args.market_impact,
        dynamic_rules=not args.no_dynamic,
        advance_interest_rate=0.12,
        auto_close_at_end=True,
        allow_odd_lot=args.allow_odd_lot,
        max_volume_ratio=args.max_vol_ratio,
        adjust_corporate_actions=args.adjust_corp_actions,
        force_adjusted=None if args.force_adjusted == "auto" else (args.force_adjusted == "true"),
        margin_ratio=args.margin_ratio,
        margin_interest_rate=args.margin_interest,
        margin_maintenance_ratio=args.margin_maintenance,
        strategy_params=strat_params,
        rights_listing_delay=args.rights_listing_delay
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
    if args.report_name:
        report_filename = args.report_name
    else:
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
