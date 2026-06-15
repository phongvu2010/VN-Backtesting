import os
import sys
import subprocess
import glob
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template

app = Flask(__name__, template_folder='templates')

# Track running process globally
# structure: {"proc": Popen_object, "start_time": datetime, "ticker": str}
backtest_process = None

# Set working directory to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT_DIR, 'reports'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'data_cache'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """Scan the cache directory and return a list of available cached tickers."""
    cache_path = os.path.join(ROOT_DIR, 'data_cache', 'equity_*.csv')
    files = glob.glob(cache_path)
    tickers = []
    for f in files:
        basename = os.path.basename(f)
        ticker = basename.replace('equity_', '').replace('.csv', '')
        if ticker:
            tickers.append(ticker)
    
    # Add some popular defaults if cache is empty
    defaults = ['FPT', 'HPG', 'VNM', 'BSR', 'IDC', 'PVS', 'MWG', 'TCB']
    for d in defaults:
        if d not in tickers:
            tickers.append(d)
            
    return jsonify(sorted(tickers))

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """List all HTML reports in reports directory, sorted by modification time descending."""
    reports_path = os.path.join(ROOT_DIR, 'reports', 'report_*.html')
    files = glob.glob(reports_path)
    
    # Sort files by modification date
    files.sort(key=os.path.getmtime, reverse=True)
    
    reports = []
    for f in files:
        basename = os.path.basename(f)
        # format: report_portfolio_YYYYMMDD_HHMMSS.html or report_opt_YYYYMMDD_HHMMSS.html
        time_str = "Unknown"
        display_name = basename
        
        try:
            name_no_ext = basename.replace('.html', '')
            parts = name_no_ext.split('_')
            if len(parts) >= 2:
                dt_str = parts[-2] + parts[-1] # YYYYMMDDHHMMSS
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                time_str = dt.strftime("%d/%m/%Y %H:%M:%S")
                
                if 'report_portfolio_' in basename:
                    display_name = f"Báo cáo Danh mục ({time_str})"
                elif 'report_opt_' in basename:
                    display_name = f"Báo cáo Tối ưu hóa ({time_str})"
        except Exception:
            pass
            
        reports.append({
            'filename': basename,
            'display_name': display_name,
            'time': time_str,
            'size_kb': round(os.path.getsize(f) / 1024.0, 1)
        })
        
    return jsonify(reports)

@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serve the static report HTML files."""
    return send_from_directory(os.path.join(ROOT_DIR, 'reports'), filename)

@app.route('/api/run', methods=['POST'])
def run_backtest():
    """Start the backtest script asynchronously."""
    global backtest_process
    
    # Check if a process is already running
    if backtest_process is not None and backtest_process["proc"].poll() is None:
        return jsonify({"error": "Một tiến trình Backtest khác đang chạy. Vui lòng đợi."}), 400
        
    data = request.json or {}
    ticker = data.get('ticker', 'FPT').strip()
    start = data.get('start', '2020-01-01')
    end = data.get('end', '2026-06-01')
    cash = data.get('cash', 100000000.0)
    rebalance_interval = data.get('rebalance_interval')
    stop_loss = data.get('stop_loss')
    trailing_stop = data.get('trailing_stop')
    margin_ratio = data.get('margin_ratio', 1.0)
    optimize = data.get('optimize', False)
    execution_at = data.get('execution_at', 'open').strip()
    slippage = data.get('slippage')
    market_impact = data.get('market_impact')
    
    # Determine python execution path in .venv
    if sys.platform == "win32":
        python_path = os.path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe')
    else:
        python_path = os.path.join(ROOT_DIR, '.venv', 'bin', 'python')
        
    if not os.path.exists(python_path):
        python_path = "python3" # Fallback if venv is not found
        
    cmd = [python_path, 'run_backtest.py', '--ticker', ticker, '--start', start, '--end', end, '--cash', str(cash)]
    
    if rebalance_interval is not None and str(rebalance_interval).strip() != "":
        cmd.extend(['--rebalance_interval', str(rebalance_interval)])
        
    if stop_loss is not None and str(stop_loss).strip() != "":
        # Convert e.g. 7% -> 0.07
        try:
            val = float(stop_loss)
            if val > 1.0:
                val = val / 100.0
            cmd.extend(['--stop_loss', str(val)])
        except ValueError:
            pass
            
    if trailing_stop is not None and str(trailing_stop).strip() != "":
        try:
            val = float(trailing_stop)
            if val > 1.0:
                val = val / 100.0
            cmd.extend(['--trailing_stop', str(val)])
        except ValueError:
            pass
            
    if margin_ratio is not None and str(margin_ratio).strip() != "":
        try:
            val = float(margin_ratio)
            cmd.extend(['--margin_ratio', str(val)])
        except ValueError:
            pass
            
    if execution_at:
        cmd.extend(['--execution_at', execution_at])
        
    if slippage is not None and str(slippage).strip() != "":
        try:
            val = float(slippage)
            if val > 1.0:
                val = val / 100.0
            cmd.extend(['--slippage', str(val)])
        except ValueError:
            pass
            
    if market_impact is not None and str(market_impact).strip() != "":
        try:
            val = float(market_impact)
            cmd.extend(['--market_impact', str(val)])
        except ValueError:
            pass
            
    if optimize:
        cmd.append('--optimize')
        
    log_file_path = os.path.join(ROOT_DIR, 'reports', 'last_run.log')
    
    # Write starting command to log
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== KHỞI CHẠY TIẾN TRÌNH BACKTEST ===\n")
        f.write(f"Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Lệnh thực thi: {' '.join(cmd)}\n")
        f.write(f"=====================================\n\n")
        
    # Open log file for subprocess output redirection
    log_file = open(log_file_path, 'a', encoding='utf-8')
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=ROOT_DIR,
            text=True
        )
        
        backtest_process = {
            "proc": proc,
            "start_time": datetime.now(),
            "ticker": ticker,
            "optimize": optimize,
            "log_file": log_file
        }
        
        return jsonify({"status": "started", "ticker": ticker})
    except Exception as e:
        log_file.close()
        return jsonify({"error": f"Không thể khởi chạy tiến trình: {e}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the running status and return the generated report file if completed."""
    global backtest_process
    
    if backtest_process is None:
        return jsonify({"status": "idle"})
        
    proc = backtest_process["proc"]
    poll = proc.poll()
    
    if poll is None:
        return jsonify({"status": "running", "ticker": backtest_process["ticker"]})
    
    # Close log file descriptor
    if "log_file" in backtest_process and backtest_process["log_file"]:
        try:
            backtest_process["log_file"].close()
        except Exception:
            pass
            
    if poll == 0:
        # Search for the latest report generated
        reports_path = os.path.join(ROOT_DIR, 'reports', 'report_*.html')
        files = glob.glob(reports_path)
        
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            latest_report = os.path.basename(files[0])
            # Verify if this report is newer than the process start time
            mtime = datetime.fromtimestamp(os.path.getmtime(files[0]))
            if mtime >= backtest_process["start_time"] or len(files) == 1:
                return jsonify({
                    "status": "completed", 
                    "ticker": backtest_process["ticker"], 
                    "report": latest_report,
                    "optimize": backtest_process["optimize"]
                })
                
        return jsonify({"status": "completed", "ticker": backtest_process["ticker"], "note": "Không tìm thấy file báo cáo mới sinh ra.", "optimize": backtest_process["optimize"]})
    else:
        return jsonify({"status": "failed", "ticker": backtest_process["ticker"], "error_code": poll})

@app.route('/api/log', methods=['GET'])
def get_log():
    """Return the current logs of the running or last run backtest."""
    log_file_path = os.path.join(ROOT_DIR, 'reports', 'last_run.log')
    if not os.path.exists(log_file_path):
        return ""
        
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Lỗi đọc file log: {e}"

if __name__ == '__main__':
    print("=" * 60)
    print("KHỞI CHẠY SERVER WEB VN-BACKTEST")
    print("URL truy cập: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
