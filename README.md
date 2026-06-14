# Hướng dẫn Xử lý & Kết quả Hệ thống VN-Backtest

Chúng tôi đã xây dựng thành công hệ thống Backtesting chuyên dụng cho thị trường chứng khoán Việt Nam (VN-Backtest). Hệ thống này đã được triển khai hoàn chỉnh dưới dạng gói Python cục bộ, có khả năng tải dữ liệu thực tế thông qua thư viện `vnstock` và mô phỏng chính xác các quy tắc giao dịch đặc thù của Việt Nam.

## Các Thay Đổi Đã Thực Hiện

Chúng tôi đã tạo ra cấu trúc mô-đun hóa chuyên nghiệp cho dự án:

1. **[requirements.txt](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/requirements.txt)**: Thêm các gói thư viện phụ thuộc (`vnstock`, `pandas`, `numpy`, `plotly`, `jinja2`).
2. **[vn_backtest/__init__.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/__init__.py)**: Điểm khởi tạo package, xuất các lớp API cốt lõi.
3. **[vn_backtest/data.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/data.py)**: Bộ tải và quản lý dữ liệu lịch sử với tính năng lưu cache cục bộ dạng CSV trong thư mục `data_cache/` nhằm tối ưu tốc độ.
4. **[vn_backtest/strategy.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/strategy.py)**: Lớp Chiến lược cơ sở (`Strategy`) cung cấp các API giao dịch và các chỉ báo kỹ thuật cơ bản như SMA, EMA, RSI, MACD, Bollinger Bands mà không phụ thuộc vào thư viện bên ngoài.
5. **[vn_backtest/engine.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/engine.py)**: Bộ mô phỏng giao dịch cốt lõi (Event-driven):
   - Quản lý quy trình thanh toán cổ phiếu **T+1.5 / T+2** dựa trên chỉ số ngày giao dịch thực tế từ dữ liệu giá (tự động bỏ qua ngày cuối tuần, lễ Tết).
   - Ràng buộc khớp lệnh theo **Lô tối thiểu 100 cổ phiếu**.
   - Áp dụng phí giao dịch mua/bán và **thuế bán 0.1%** đặc thù của Việt Nam.
   - Kiểm tra **biên độ trần/sàn** dựa trên sàn giao dịch tương ứng (HOSE: +/-7%, HNX: +/-10%, UPCOM: +/-15%) và chặn giao dịch khi xuất hiện khóa trần (Ceiling Buy Lock) hoặc khóa sàn (Floor Sell Lock).
6. **[vn_backtest/analysis.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/analysis.py)**: Bộ phân tích hiệu suất đầu tư, tính toán CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, cũng như so sánh hiệu quả với benchmark VN-Index (Alpha, Beta, Outperformance).
7. **[vn_backtest/reporter.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/reporter.py)**: Bộ tạo báo cáo HTML cao cấp với giao diện Dark-mode và đồ thị Plotly tương tác.
8. **[vn_backtest/strategies/ma_cross.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/vn_backtest/strategies/ma_cross.py)**: Chiến lược mẫu Moving Average Crossover (đường SMA 10 cắt SMA 20).
9. **[run_backtest.py](file:///Users/hunterdo/Documents/Python%20Project/Backtesting-Trading/run_backtest.py)**: Giao diện dòng lệnh CLI chạy trực tiếp chương trình kiểm thử.

---

## Kết Quả Xác Thực & Kiểm Thử

Chúng tôi đã thực hiện 2 hình thức kiểm thử độc lập để đảm bảo chất lượng hệ thống:

### 1. Kiểm thử Logic Tự động (Unit/Logic Verification)
Chúng tôi đã chạy kịch bản kiểm thử trong [test_t2.py](file:///Users/hunterdo/.gemini/antigravity/brain/0eb39485-48dc-4977-9c3f-d02572292cbd/scratch/test_t2.py) trên 5 ngày dữ liệu giả lập để kiểm tra độ chính xác của các quy tắc:
* **Khớp lệnh mua & Làm tròn lô:** Mua tối đa với vốn 10.000.000 VND tại giá 10 (phí mua 0.15%). Hệ thống tính toán chính xác và khớp 998,500 cổ phiếu (làm tròn xuống từ 998,502.2), giữ lại số dư tiền mặt lẻ. -> **ĐẠT**
* **Khóa T+2:** Đặt lệnh bán 100 cổ phiếu vào ngày T+1 (sau ngày mua 1 ngày). Hệ thống tự động từ chối và ghi nhận lý do `No sellable shares`. -> **ĐẠT**
* **Giải tỏa T+2:** Đến ngày T+3 (đủ 2 ngày giao dịch chờ thanh toán kể từ ngày mua T+1), đặt lệnh bán toàn bộ. Hệ thống giải tỏa thành công và khớp bán toàn bộ 998,500 cổ phiếu. -> **ĐẠT**
* **Thuế & Phí bán:** Lệnh bán được trừ phí môi giới 0.15% và thuế TNCN 0.1% trực tiếp vào giá trị thực nhận. -> **ĐẠT**

### 2. Kiểm thử Thực tế trên Cổ phiếu FPT (2020 - 2026)
Chúng tôi đã tiến hành chạy thử thực tế với cổ phiếu **FPT** từ `2020-01-01` đến `2026-06-01` sử dụng chiến lược giao cắt SMA 10/20.

**Kết quả tóm tắt:**
* **Vốn ban đầu**: 100,000,000 VND
* **Tài sản cuối kỳ**: 208,260,445 VND
* **Tổng lợi nhuận**: **108.26%** (so với VN-Index cùng kỳ tăng 90.81%)
* **Lợi nhuận năm (CAGR)**: **12.12%**
* **Hệ số Sharpe**: 0.41
* **Mức sụt giảm lớn nhất (MDD)**: -23.25%
* **Tổng số giao dịch**: 79 lệnh
* **Tỷ lệ thắng (Win Rate)**: 41.0%
* **Hiệu số Alpha**: +5.48% (chỉ số Alpha vượt trội so với thị trường chung)
* **Hệ số Beta**: 0.40 (mức độ biến động thấp hơn nhiều so với thị trường chung, chứng tỏ rủi ro hệ thống thấp)

Báo cáo tương tác HTML với đầy đủ biểu đồ tăng trưởng tài sản (Equity Curve), mức sụt giảm (Drawdown) và các điểm mua bán trực quan đã được lưu tại:
`reports/report_FPT_20260615_041732.html`

---

## Hướng Dẫn Sử Dụng Hệ Thống

Để khởi chạy backtest cho bất kỳ cổ phiếu nào trên thị trường chứng khoán Việt Nam, bạn có thể sử dụng các tham số CLI linh hoạt của `run_backtest.py`:

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Chạy cấu hình mặc định (Mã FPT từ 2020 đến 2026)
python run_backtest.py

# Chạy backtest với mã khác, ví dụ HPG từ năm 2022 đến 2026
python run_backtest.py --ticker HPG --start 2022-01-01 --end 2026-06-01 --cash 200000000

# Chạy backtest với mã sàn HNX (biên độ +/-10%), ví dụ IDC
python run_backtest.py --ticker IDC --start 2021-01-01 --end 2026-06-01 --exchange hnx
```

Các tham số CLI khả dụng:
* `--ticker`: Mã cổ phiếu (FPT, HPG, VCB, v.v.)
* `--start` & `--end`: Khoảng thời gian test (`YYYY-MM-DD`)
* `--cash`: Vốn ban đầu (mặc định: 100M VND)
* `--exchange`: Tên sàn để áp trần/sàn (`hose`, `hnx`, `upcom`)
* `--t_settle`: Chu kỳ thanh toán (mặc định: 2, đại diện cho T+2)
* `--lot_size`: Lô giao dịch tối thiểu (mặc định: 100)
* `--fee` & `--tax`: Phí môi giới và thuế bán (mặc định tương ứng: 0.15% và 0.1%)
