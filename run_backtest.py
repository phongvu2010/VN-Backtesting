import os
import argparse
from datetime import datetime, timedelta
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
    parser.add_argument("--exchange", type=str, default="hose", choices=["hose", "hnx", "upcom"], help="Sàn giao dịch để áp biên độ trần/sàn (mặc định: hose)")
    parser.add_argument("--t_settle", type=int, default=2, help="Chu kỳ thanh toán cổ phiếu (mặc định: T+2)")
    parser.add_argument("--lot_size", type=int, default=100, help="Lô giao dịch tối thiểu (mặc định: 100)")
    parser.add_argument("--fee", type=float, default=0.0015, help="Phí giao dịch mua/bán (mặc định: 0.15%)")
    parser.add_argument("--tax", type=float, default=0.001, help="Thuế bán chứng khoán (mặc định: 0.1%)")
    parser.add_argument("--no_cache", action="store_true", help="Không sử dụng cache dữ liệu, tải mới hoàn toàn")
    parser.add_argument("--no_dynamic", action="store_true", help="Vô hiệu hóa quy tắc lịch sử động (chạy tĩnh)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"KHỞI CHẠY HỆ THỐNG VN-BACKTEST CHO MÃ: {args.ticker.upper()}")
    print(f"Thời gian: {args.start} -> {args.end}")
    print(f"Vốn ban đầu: {args.cash:,.0f} VND")
    if not args.no_dynamic:
        print(f"Cấu hình: Sàn {args.exchange.upper()} (Biên độ trần/sàn), Quy tắc lịch sử động (T+2.5/T+3 & Lô 1/10/100)")
    else:
        print(f"Cấu hình: Sàn {args.exchange.upper()} (Biên độ trần/sàn), T+{args.t_settle}, Lô {args.lot_size} (Cấu hình tĩnh)")
    print(f"Chi phí: Phí GD {args.fee*100:.2f}%, Thuế bán {args.tax*100:.2f}%")
    print("=" * 60)
    
    # 1. Load data
    loader = VNStockDataLoader()
    
    print("\n[1/4] Đang tải dữ liệu cổ phiếu...")
    try:
        stock_data = loader.fetch_data(
            symbol=args.ticker, 
            start_date=args.start, 
            end_date=args.end, 
            is_index=False, 
            use_cache=not args.no_cache
        )
        print(f"-> Đã tải {len(stock_data)} phiên giao dịch cổ phiếu {args.ticker.upper()}.")
    except Exception as e:
        print(f"LỖI: Không thể tải dữ liệu cho mã {args.ticker}: {e}")
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
        exchange=args.exchange,
        execution_at="open",
        restrict_ceiling_buy=True,
        restrict_floor_sell=True,
        slippage=0.0,
        dynamic_rules=not args.no_dynamic
    )
    # Inject ticker information into the engine instance so the strategy can access it
    engine.ticker = args.ticker.upper()
    
    results = engine.run()
    
    # 3. Analyze performance
    print("[4/4] Đang tính toán các chỉ số hiệu suất...")
    metrics = PerformanceAnalyzer.calculate_metrics(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        benchmark_data=benchmark_data,
        initial_cash=results['initial_cash']
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
    report_filename = f"report_{args.ticker.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    report_path = reporter.generate_report(
        metrics=metrics,
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        stock_data=stock_data,
        ticker=args.ticker.upper(),
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
