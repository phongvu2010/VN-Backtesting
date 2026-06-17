import itertools
import random
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Type, Optional
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


class SmartOptimizer:
    """
    Smart Parameter Optimizer supporting Optuna Bayesian optimization
    and a custom Genetic Algorithm.
    """

    def __init__(
        self,
        data,
        strategy_class,
        param_space: dict,
        initial_cash: float = 100_000_000.0,
        exchange="hose",
        benchmark_data=None,
        risk_free_rate: float = 0.04,
        engine_kwargs: dict = None,
        method: str = 'optuna',
        n_trials: int = 100,
        timeout: int = None,
        population_size: int = 50,
        n_generations: int = 20,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        n_jobs: int = -1,
    ):
        self.data = data
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.initial_cash = initial_cash
        self.exchange = exchange
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        self.engine_kwargs = engine_kwargs or {}
        self.method = method.lower()
        self.n_trials = n_trials
        self.timeout = timeout
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_jobs = n_jobs

        # Internal state
        self._study = None  # Optuna study (if used)
        self._results: List[dict] = []
        self._best_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Helper: evaluate a single parameter set
    # ------------------------------------------------------------------
    def _evaluate(self, params: dict, idx: int = 0) -> Optional[dict]:
        """Run a single backtest and return the record dict, or None on error."""
        task_args = (
            idx,
            params,
            self.data,
            self.strategy_class,
            self.initial_cash,
            self.exchange,
            self.benchmark_data,
            self.risk_free_rate,
            self.engine_kwargs,
        )
        record, err = _run_single_backtest(task_args)
        if err:
            print(f"   LỖI khi chạy tham số {err[0]}: {err[1]}")
            return None
        return record

    # ------------------------------------------------------------------
    # Helper: determine param type from space definition
    # ------------------------------------------------------------------
    @staticmethod
    def _is_categorical(spec) -> bool:
        """Return True when spec is a list of choices (categorical / discrete)."""
        return isinstance(spec, list)

    @staticmethod
    def _is_int_range(spec) -> bool:
        """Return True when spec is a (min, max) tuple of ints."""
        return (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[0], int)
            and isinstance(spec[1], int)
        )

    @staticmethod
    def _is_float_range(spec) -> bool:
        """Return True when spec is a (min, max) tuple containing at least one float."""
        return (
            isinstance(spec, tuple)
            and len(spec) == 2
            and any(isinstance(v, float) for v in spec)
        )

    # ------------------------------------------------------------------
    # Optuna Bayesian Optimization
    # ------------------------------------------------------------------
    def _run_optuna(self, metric: str, ascending: bool) -> pd.DataFrame:
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import MedianPruner
        except ImportError:
            raise ImportError(
                "Optuna chưa được cài đặt. Hãy chạy: pip install optuna"
            )

        direction = 'minimize' if ascending else 'maximize'
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )
        # Suppress Optuna's own logging to keep output clean
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        results: List[dict] = []
        total = self.n_trials

        print("=" * 60)
        print(f"KHỞI CHẠY TỐI ƯU HÓA THÔNG MINH (optuna) - {total} trials")
        print("=" * 60)

        def objective(trial: 'optuna.Trial') -> float:
            params: dict = {}
            for name, spec in self.param_space.items():
                if self._is_categorical(spec):
                    params[name] = trial.suggest_categorical(name, spec)
                elif self._is_int_range(spec):
                    params[name] = trial.suggest_int(name, spec[0], spec[1])
                elif self._is_float_range(spec):
                    params[name] = trial.suggest_float(name, float(spec[0]), float(spec[1]))
                else:
                    raise ValueError(f"Không hỗ trợ kiểu param_space cho '{name}': {spec}")

            record = self._evaluate(params, idx=trial.number + 1)
            if record is None:
                # Return worst possible value so Optuna continues
                return float('-inf') if direction == 'maximize' else float('inf')

            results.append(record)
            value = record.get(metric, 0.0)
            print(
                f"[{trial.number + 1}/{total}] Trial #{trial.number + 1}: "
                f"{metric}={value:.4f}, Params={params}"
            )
            return value

        study.optimize(
            objective,
            n_trials=total,
            timeout=self.timeout,
            n_jobs=1,  # sequential inside optimize; parallelism handled externally if needed
            show_progress_bar=False,
        )

        self._study = study
        self._results = results

        if not results:
            print("❌ Không có trial nào thành công.")
            return pd.DataFrame()

        self._best_params = study.best_params

        results_df = pd.DataFrame(results)
        results_df.sort_values(by=metric, ascending=ascending, inplace=True)

        self._print_summary(results_df, metric)
        return results_df

    # ------------------------------------------------------------------
    # Genetic Algorithm
    # ------------------------------------------------------------------
    def _random_individual(self) -> dict:
        """Create a random individual from param_space."""
        ind: dict = {}
        for name, spec in self.param_space.items():
            if self._is_categorical(spec):
                ind[name] = random.choice(spec)
            elif self._is_int_range(spec):
                ind[name] = random.randint(spec[0], spec[1])
            elif self._is_float_range(spec):
                ind[name] = random.uniform(float(spec[0]), float(spec[1]))
            else:
                raise ValueError(f"Không hỗ trợ kiểu param_space cho '{name}': {spec}")
        return ind

    def _tournament_select(self, population: List[dict], fitnesses: List[float], k: int = 3) -> dict:
        """Tournament selection: pick the best among *k* random individuals."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """Uniform crossover."""
        child: dict = {}
        for name in self.param_space:
            child[name] = parent1[name] if random.random() < 0.5 else parent2[name]
        return child

    def _mutate(self, individual: dict) -> dict:
        """Random mutation: for each gene, re-randomize with probability mutation_rate."""
        for name, spec in self.param_space.items():
            if random.random() < self.mutation_rate:
                if self._is_categorical(spec):
                    individual[name] = random.choice(spec)
                elif self._is_int_range(spec):
                    individual[name] = random.randint(spec[0], spec[1])
                elif self._is_float_range(spec):
                    individual[name] = random.uniform(float(spec[0]), float(spec[1]))
        return individual

    def _run_genetic(self, metric: str, ascending: bool) -> pd.DataFrame:
        pop_size = self.population_size
        n_gen = self.n_generations

        print("=" * 60)
        print(f"KHỞI CHẠY TỐI ƯU HÓA THÔNG MINH (genetic) - {pop_size * n_gen} trials")
        print("=" * 60)

        all_results: List[dict] = []
        trial_counter = 0

        # --- Initialize population ---
        population = [self._random_individual() for _ in range(pop_size)]

        # Evaluate initial population
        fitnesses: List[float] = []
        for ind in population:
            trial_counter += 1
            record = self._evaluate(ind, idx=trial_counter)
            if record is not None:
                all_results.append(record)
                val = record.get(metric, 0.0)
                if ascending:
                    val = -val  # GA maximizes; invert for ascending (minimize)
                fitnesses.append(val)
            else:
                fitnesses.append(float('-inf'))

        best_fitness = max(fitnesses)
        print(f"Thế hệ 0/{n_gen}: Best Fitness = {best_fitness:.4f}")

        for gen in range(1, n_gen + 1):
            new_population: List[dict] = []
            new_fitnesses: List[float] = []

            for _ in range(pop_size):
                # Selection
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1

                # Mutation
                child = self._mutate(child)

                # Evaluate
                trial_counter += 1
                record = self._evaluate(child, idx=trial_counter)
                if record is not None:
                    all_results.append(record)
                    val = record.get(metric, 0.0)
                    if ascending:
                        val = -val
                    new_fitnesses.append(val)
                else:
                    new_fitnesses.append(float('-inf'))

                new_population.append(child)

            population = new_population
            fitnesses = new_fitnesses

            gen_best = max(fitnesses)
            if gen_best > best_fitness:
                best_fitness = gen_best
            print(f"Thế hệ {gen}/{n_gen}: Best Fitness = {best_fitness:.4f}")

        self._results = all_results

        if not all_results:
            print("❌ Không có trial nào thành công.")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        results_df.sort_values(by=metric, ascending=ascending, inplace=True)

        # Store best params
        best_row = results_df.iloc[0]
        self._best_params = {
            name: best_row[name]
            for name in self.param_space
            if name in best_row.index
        }

        self._print_summary(results_df, metric)
        return results_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_optimization(self, metric: str = 'sharpe_ratio', ascending: bool = False) -> pd.DataFrame:
        """Run smart optimization. Returns DataFrame with same columns as ParameterOptimizer."""
        if self.method == 'optuna':
            return self._run_optuna(metric, ascending)
        elif self.method == 'genetic':
            return self._run_genetic(metric, ascending)
        else:
            raise ValueError(
                f"Phương pháp không hợp lệ: '{self.method}'. Chọn 'optuna' hoặc 'genetic'."
            )

    def get_best_params(self) -> dict:
        """Get the best parameters found."""
        if self._best_params is None:
            raise RuntimeError("Chưa chạy tối ưu hóa. Hãy gọi run_optimization() trước.")
        return self._best_params

    def get_param_importance(self) -> dict:
        """Get parameter importance ranking (Optuna only)."""
        if self._study is None:
            raise RuntimeError(
                "Chức năng này chỉ hỗ trợ phương pháp 'optuna'. "
                "Hãy chạy run_optimization() với method='optuna' trước."
            )
        try:
            import optuna
            importance = optuna.importance.get_param_importances(self._study)
            return importance
        except ImportError:
            raise ImportError(
                "Optuna chưa được cài đặt. Hãy chạy: pip install optuna"
            )
        except Exception as e:
            print(f"Không thể tính toán parameter importance: {e}")
            return {}

    # ------------------------------------------------------------------
    # Pretty-print summary
    # ------------------------------------------------------------------
    def _print_summary(self, results_df: pd.DataFrame, metric: str) -> None:
        print("\n" + "=" * 60)
        print(" TỐI ƯU HÓA HOÀN TẤT - BẢNG XẾP HẠNG THAM SỐ")
        print("=" * 60)

        top_n = min(5, len(results_df))
        temp_df = results_df.copy()

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
