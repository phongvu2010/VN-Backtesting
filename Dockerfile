# Sử dụng Python 3.12 phiên bản rút gọn (slim) để nhẹ nhất có thể
FROM python:3.12-slim

# Ngăn Python tạo file cache và ép xuất log ra console ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Tạo một tài khoản người dùng phi đặc quyền (non-root) tên là 'appuser'
RUN useradd -m appuser

# 2. Chuyển quyền điều khiển sang 'appuser' và tạo thư mục làm việc
USER appuser
WORKDIR /home/appuser/app

# 3. Tạo môi trường ảo (Virtual Environment) cho appuser
ENV VIRTUAL_ENV=/home/appuser/venv
RUN python -m venv $VIRTUAL_ENV

# 4. Đưa môi trường ảo lên đầu biến PATH để hệ thống mặc định dùng nó
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. Copy file requirements (cấp quyền cho appuser) và cài đặt thư viện
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ mã nguồn vào container
COPY --chown=appuser:appuser . .

# 7. Khai báo cổng mà ứng dụng sẽ lắng nghe bên trong Container
EXPOSE 5000

# 8. Khai báo lệnh thực thi bằng Gunicorn thay vì python trực tiếp
# Bắt buộc cấu hình --workers 1 và --threads 4 để bảo toàn biến toàn cục running_tasks
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "web_app:app"]
