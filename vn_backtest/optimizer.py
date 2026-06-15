import itertools
import pandas as pd
from typing import Dict, List, Any, Type
from .engine import BacktestEngine
from .strategy import Strategy
from .analysis import PerformanceAnalyzer

def _run_single_backtest(task_args):
    """Helper function to run a single backtest combination. Must be top-level for multiprocessing."""
    idx, params, data, strategy_class, initial_cash, exchange, benchmark_data, risk_free_rate, engine_kwargs = task_args
    from vn_backtest.engine import BacktestEngine
    from vn_backtest.analysis import PerformanceAnalyzer
    import traceback
    
    try:
        # Merge custom engine kwargs with the current strategy parameters
        kwargs = engine_kwargs.copy()
        
        # Merge global strategy parameters (e.g. SL/TS from CLI) with grid parameters
        combined_strategy_params = params.copy()
        if 'strategy_params' in kwargs:
            cli_strat_params = kwargs.pop('strategy_params')
            if cli_strat_params:
                combined_strategy_params.update(cli_strat_params)
        
        # Instantiate engine
        engine = BacktestEngine(
            data=data,
            strategy_class=strategy_class,
            initial_cash=initial_cash,
            exchange=exchange,
            strategy_params=combined_strategy_params,
            **kwargs
        )
        
        # Run backtest
        backtest_res = engine.run()
        
        # Calculate metrics
        metrics = PerformanceAnalyzer.calculate_metrics(
            equity_curve=backtest_res['equity_curve'],
            trades=backtest_res['trades'],
            benchmark_data=benchmark_data,
            initial_cash=initial_cash,
            risk_free_rate=risk_free_rate,
            include_auto_close=True
        )
        
        # Combine parameters and performance metrics
        record = params.copy()
        record.update({
            'total_return': metrics.get('total_return', 0.0),
            'cagr': metrics.get('cagr', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'sortino_ratio': metrics.get('sortino_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'total_trades': metrics.get('total_trades', 0)
        })
        return record, None
    except Exception as e:
        err_msg = f"{e}\n{traceback.format_exc()}"
        return None, (params, err_msg)

class ParameterOptimizer:
    """
    Parameter Optimizer for Grid Searching strategy parameters.
    Runs multiple backtest configurations and compiles performance metrics.
    """
    def __init__(
        self,
        data: Any,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        initial_cash: float = 100_000_000.0,
        exchange: Any = "hose",
        benchmark_data: pd.DataFrame = None,
        risk_free_rate: float = 0.04,
        engine_kwargs: Dict[str, Any] = None,
        n_jobs: int = -1
    ):
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.initial_cash = initial_cash
        self.exchange = exchange
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        self.engine_kwargs = engine_kwargs or {}
        self.n_jobs = n_jobs

    def run_optimization(self, sort_by: str = "sharpe_ratio", ascending: bool = False) -> pd.DataFrame:
        """
        Run the grid search optimization over all parameter combinations.
        
        Args:
            sort_by (str): Metric to sort results by (e.g. 'sharpe_ratio', 'total_return', 'cagr', 'max_drawdown').
            ascending (bool): True to sort ascending, False to sort descending.
            
        Returns:
            pd.DataFrame: Optimization results with parameters and metrics.
        """
        # Generate parameter combinations
        keys, values = zip(*self.param_grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print("=" * 60)
        print(f"KHỞI CHẠY TỐI ƯU HÓA THAM SỐ (Tổng cộng {len(permutations)} tổ hợp)")
        print("=" * 60)
        
        results = []
        
        if self.n_jobs == 1:
            # Run sequentially
            for idx, params in enumerate(permutations, 1):
                print(f"[{idx}/{len(permutations)}] Đang chạy thử nghiệm với tham số (tuần tự): {params}...")
                task_args = (idx, params, self.data, self.strategy_class, self.initial_cash, self.exchange,
                             self.benchmark_data, self.risk_free_rate, self.engine_kwargs)
                record, err = _run_single_backtest(task_args)
                if err:
                    print(f"   LỖI khi chạy tổ hợp {err[0]}: {err[1]}")
                else:
                    results.append(record)
        else:
            # Run in parallel
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            n_workers = self.n_jobs if self.n_jobs > 0 else os.cpu_count() or 1
            print(f"-> Chạy song song sử dụng {n_workers} tiến trình...")
            
            # Prepare task arguments
            tasks = [
                (idx, params, self.data, self.strategy_class, self.initial_cash, self.exchange,
                 self.benchmark_data, self.risk_free_rate, self.engine_kwargs)
                for idx, params in enumerate(permutations, 1)
            ]
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all combinations to the executor
                futures = {executor.submit(_run_single_backtest, task): task for task in tasks}
                
                for future in as_completed(futures):
                    task = futures[future]
                    params = task[1]
                    try:
                        record, err = future.result()
                        if err:
                            print(f"   LỖI khi chạy tổ hợp {params}: {err[1]}")
                        else:
                            print(f"[{len(results)+1}/{len(permutations)}] Đã hoàn thành tham số: {record}")
                            results.append(record)
                    except Exception as e:
                        print(f"   LỖI hệ thống khi chạy tổ hợp {params}: {e}")
                        
        if not results:
            print("❌ Không chạy thành công bất kỳ tổ hợp tham số nào.")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        # Sort results
        if sort_by in results_df.columns:
            results_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
            
        print("\n" + "=" * 60)
        print(" TỐI ƯU HÓA HOÀN TẤT - BẢNG XẾP HẠNG THAM SỐ")
        print("=" * 60)
        
        # Display top 5 parameter sets
        top_n = min(5, len(results_df))
        temp_df = results_df.copy()
        
        # Formatter for pretty printing
        pct_cols = ['total_return', 'cagr', 'max_drawdown', 'win_rate']
        for col in pct_cols:
            if col in temp_df.columns:
                temp_df[col] = (temp_df[col] * 100).map('{:.2f}%'.format)
                
        ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        for col in ratio_cols:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].map('{:.2f}'.format)
                
        print(temp_df.head(top_n).to_string(index=False))
        print("=" * 60 + "\n")
        
        return results_df
