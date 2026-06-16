# Hướng dẫn Xử lý & Kết quả Hệ thống VN-Backtest

Chúng tôi đã xây dựng thành công hệ thống Backtesting chuyên dụng cho thị trường chứng khoán Việt Nam (VN-Backtest). Hệ thống này hỗ trợ kiểm thử các chiến lược giao dịch đa tài sản (Portfolio), tự động tải dữ liệu thực tế thông qua thư viện `vnstock` và mô phỏng chính xác các quy tắc tài chính, giao dịch đặc thù của Việt Nam.

---

## Các Tính Năng & Thay Đổi Cốt Lõi

Hệ thống được thiết kế theo dạng mô-đun hóa chuyên nghiệp và đã được nâng cấp toàn diện:

1. **[strategy.py](file:///Users/hunterdo/Documents/Python%20Project/VN%20Backtesting/vn_backtest/strategy.py)**: Lớp Chiến lược cơ sở (`Strategy`) hỗ trợ giao dịch đa tài sản. Cung cấp các hàm getter động (`get_open()`, `get_close()`, v.v.) và cơ chế tính toán chỉ báo kỹ thuật riêng biệt cho từng cổ phiếu qua hàm `self.I(indicator_func, ticker, *args)`.
2. **[engine.py](file:///Users/hunterdo/Documents/Python%20Project/VN%20Backtesting/vn_backtest/engine.py)**: Động cơ mô phỏng giao dịch cốt lõi (Event-driven):
   * **Đồng bộ hóa dòng thời gian đa tài sản**: Tự động ghép nối dữ liệu lịch sử của nhiều mã cổ phiếu thành một dòng thời gian chung (`dates`) để chạy mô phỏng.
   * **Chu kỳ thanh toán Cổ phiếu & Tiền mặt (T+2)**: Mô phỏng chu kỳ thanh toán T+1.5/T+2 cho cả cổ phiếu và tiền bán chờ về.
   * **Phí ứng trước tiền bán (Cash Advance Fee)**: Tự động cho phép mua cổ phiếu bằng tiền bán chờ về và tính lãi vay ứng trước danh nghĩa (mặc định $12\%/\text{năm}$, tính theo số ngày chờ thực tế) trừ thẳng vào tài sản.
   * **Làm tròn Trần/Sàn theo Tick Size**: Tự động làm tròn giá trần xuống và giá sàn lên theo đúng bước giá quy định của HOSE (10đ, 50đ, 100đ) và HNX/UPCoM (100đ).
   * **Tất toán cuối kỳ (Auto-Close)**: Tự động bán toàn bộ cổ phiếu nắm giữ ở phiên cuối cùng theo giá Close để phản ánh chính xác chi phí thuế, phí và tính toán thống kê giao dịch đầy đủ.
3. **[analysis.py](file:///Users/hunterdo/Documents/Python%20Project/VN%20Backtesting/vn_backtest/analysis.py)**: Phân tích hiệu suất đầu tư bằng thuật toán ghép cặp FIFO tách biệt theo từng mã cổ phiếu (`buy_queues`), đo lường chính xác CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, Alpha và Beta.
4. **[reporter.py](file:///Users/hunterdo/Documents/Python%20Project/VN%20Backtesting/vn_backtest/reporter.py)**: Tạo báo cáo HTML cao cấp với đồ thị Plotly tương tác hiển thị song song đường giá của nhiều mã cổ phiếu cùng nhật ký tín hiệu mua/bán động.
5. **[strategies/ma_cross.py](file:///Users/hunterdo/Documents/Python%20Project/VN%20Backtesting/vn_backtest/strategies/ma_cross.py)**: Chiến lược mẫu Moving Average Crossover (SMA 10 cắt SMA 20) được tối ưu để tự động phân bổ vốn đều và giao dịch đồng thời trên nhiều cổ phiếu.

---

## Hướng Dẫn Sử Dụng

Bạn có thể chạy backtest cho một hoặc nhiều cổ phiếu cùng lúc bằng các tham số CLI linh hoạt trong `run_backtest.py`:

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Chạy cấu hình mặc định (Mã FPT từ năm 2020 đến 2026)
python run_backtest.py

# Chạy backtest danh mục 2 mã HOSE (FPT và HPG)
python run_backtest.py --ticker FPT,HPG --start 2020-01-01 --end 2026-06-01 --cash 100000000

# Chạy backtest danh mục sàn HNX & UPCoM (IDC và BSR) bắt đầu từ năm 2021
python run_backtest.py --ticker IDC,BSR --start 2021-01-01
```

Các tham số CLI khả dụng:
* `--ticker`: Danh sách mã cổ phiếu, ngăn cách bằng dấu phẩy (ví dụ: `FPT,HPG,VNM`).
* `--start` & `--end`: Khoảng thời gian kiểm thử (`YYYY-MM-DD`).
* `--cash`: Vốn ban đầu (mặc định: 100,000,000 VND).
* `--exchange`: Tên sàn mặc định nếu không nhận diện được sàn của mã (`hose`, `hnx`, `upcom`). Hệ thống sẽ tự động đoán sàn của các mã phổ biến.
* `--t_settle`: Chu kỳ thanh toán chứng khoán tĩnh nếu tắt dynamic rules (mặc định: 2).
* `--lot_size`: Lô giao dịch tối thiểu tĩnh nếu tắt dynamic rules (mặc định: 100).
* `--fee` & `--tax`: Phí giao dịch mua/bán và thuế bán chứng khoán (mặc định tương ứng: 0.15% và 0.1%).
* `--no_cache`: Không dùng cache dữ liệu CSV cũ mà tải mới hoàn toàn từ `vnstock`.
* `--no_dynamic`: Vô hiệu hóa việc áp dụng luật thay đổi động theo dòng lịch sử (chạy cấu hình tĩnh).
* `--rebalance_interval`: Chu kỳ cơ cấu tỷ trọng danh mục theo số phiên (ví dụ: 20).
* `--n_jobs`: Số tiến trình chạy song song khi tối ưu hóa (mặc định: -1).

---

## Hướng Dẫn Khởi Chạy Ứng Dụng Web Dashboard

Hệ thống hỗ trợ giao diện Web trực quan (phong cách Glassmorphism Dark Mode) giúp cấu hình tham số, theo dõi log chạy thời gian thực và hiển thị biểu đồ báo cáo Plotly tương tác trực tiếp.

### Các bước khởi chạy:

1. **Kích hoạt virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Cài đặt thư viện bổ sung (nếu chưa cài)**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Khởi chạy Web Server**:
   ```bash
   python web_app.py
   ```

4. **Truy cập giao diện Web**:
   Mở trình duyệt và truy cập: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## Kết Quả Xác Thực & Kiểm Thử Thực Tế

Chúng tôi đã chạy thử nghiệm thực tế trong giai đoạn **2020/2021 đến 2026** với các kịch bản sau:

### Kịch bản 1: Cổ phiếu FPT (Sàn HOSE)
* **Tổng số giao dịch**: 80 lệnh (đã bao gồm 1 lệnh tất toán ở ngày cuối).
* **Vốn ban đầu**: 100,000,000 VND.
* **Tài sản cuối kỳ**: **206,351,943 VND** (Tổng lợi nhuận: **106.35%**).
* **Lợi nhuận năm (CAGR)**: **11.96%** (VN-Index cùng kỳ tăng 90.81%).
* **Hệ số Sharpe / Sortino**: 0.40 / 0.61.
* **Mức sụt giảm lớn nhất (MDD)**: -23.30%.
* **Tỷ lệ thắng (Win Rate)**: 42.5%.

### Kịch bản 2: Danh mục FPT & HPG (Sàn HOSE)
* **Tổng số giao dịch**: 160 lệnh (đã bao gồm 2 lệnh tất toán ở ngày cuối).
* **Tài sản cuối kỳ**: **190,935,047 VND** (Tổng lợi nhuận: **90.94%**).
* **Lợi nhuận năm (CAGR)**: **10.61%**.
* **Hệ số Sharpe / Sortino**: 0.43 / 0.63.
* **Mức sụt giảm lớn nhất (MDD)**: -32.98%.
* **Tỷ lệ thắng (Win Rate)**: 45.0%.

### Kịch bản 3: Danh mục IDC (Sàn HNX) & BSR (Sàn UPCoM) từ 2021
* **Tổng số giao dịch**: 134 lệnh.
* **Tài sản cuối kỳ**: **236,881,417 VND** (Tổng lợi nhuận: **136.88%**).
* **Lợi nhuận năm (CAGR)**: **17.30%** (VN-Index cùng kỳ tăng 64.62%).
* **Hệ số Sharpe / Sortino**: 0.62 / 0.93.
* **Mức sụt giảm lớn nhất (MDD)**: -32.99%.
* **Tỷ lệ thắng (Win Rate)**: 44.8%.

Báo cáo tương tác HTML với đầy đủ biểu đồ tài sản và nhật ký lệnh chi tiết được lưu tại thư mục `reports/`.

---

## Lưu ý quan trọng: Phòng tránh Look-ahead Bias khi viết chiến lược

Để đảm bảo kết quả backtest hoàn toàn thực tế và có thể áp dụng ngoài thực tế giao dịch, hệ thống **VN-Backtest** được thiết kế để triệt tiêu hiện tượng **Look-ahead Bias** (nhìn trước tương lai) thông qua mô hình khớp lệnh như sau:

1. **Cơ chế Khớp lệnh Hai Bước:**
   * Trong phương thức `next()` của chiến lược (được gọi ở cuối mỗi phiên giao dịch $T$), bạn chỉ được đặt lệnh thông qua các hàm như `self.buy()`, `self.sell()` hoặc `self.order_target_percent()`.
   * Lệnh này **sẽ không được khớp ngay lập tức** tại phiên $T$. Thay vào đó, nó được xếp vào hàng đợi `pending_orders` và chỉ được mang ra khớp ở phiên tiếp theo ($T+1$) với giá khớp được xác định bởi thuộc tính `--execution_at` (mặc định là giá `Open` của ngày $T+1$).

2. **Cách viết Chiến lược An toàn (Tránh Bias):**
   * Chỉ sử dụng dữ liệu giá lịch sử của phiên hiện tại trở về trước để đưa ra quyết định giao dịch. 
   * Tránh tuyệt đối việc lấy dữ liệu tương lai để đưa ra tín hiệu cho phiên hiện tại.
   * Ví dụ:
     * **ĐÚNG:** So sánh giá đóng cửa hôm nay (`self.close` hoặc `self.get_close(ticker)`) với chỉ báo kỹ thuật của ngày hôm nay để quyết định mua. Lệnh mua sẽ khớp vào giá Mở cửa (Open) ngày mai.
     * **SAI:** Đọc trước giá Mở cửa hoặc giá Thấp nhất của ngày mai trong khi đang ở phiên `next()` hôm nay để đặt lệnh khớp luôn cùng ngày.

