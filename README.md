# Forest_Fire_Detection

Hệ thống huấn luyện và suy luận khả năng cháy rừng dựa trên dữ liệu khí tượng (ERA5) và quan sát MODIS. Repo bao gồm pipeline xử lý dữ liệu, nhiều chiến lược huấn luyện mô hình cổ điển/lấn sâu và ứng dụng Streamlit để suy luận.

## Thư mục & nhiệm vụ từng file
| Tập tin | Nhiệm vụ chính | Hướng dẫn sử dụng |
| --- | --- | --- |
| `process_data.py` | Hợp nhất dữ liệu ERA5 & MODIS, tính lại bộ chỉ số FWI (FFMC/DMC/DC/ISI/BUI/FWI) theo lưới tọa độ và tạo nhãn `Classes`. | Đặt CSV của ERA5 trong `data/data/era5/era5_<year>.csv` và MODIS trong `data/data/nasa/modis_<year>_Vietnam.csv`, sau đó chạy `python process_data.py` để tạo `data/data/clean/clean_<year>.csv` và tập nhiều năm `clean_<start>_<end>_timelines.csv`. Điều chỉnh `BASE` và khoảng năm ở cuối file khi cần. |
| `train.py` | Huấn luyện các mô hình ML cổ điển (Random Forest, Logistic Regression, XGBoost) với pipeline chia train/val/test, cân bằng mẫu và tinh chỉnh threshold theo độ chính xác tối thiểu. Lưu model/scaler phục vụ ứng dụng suy luận. | Kiểm tra biến `CONFIG` ở hàm `main()` (đường dẫn CSV, tỷ lệ split, danh sách mô hình, thư mục log). Chạy `python train.py`; mỗi cặp mô hình–scaler tạo thư mục `logs/<model>_<scaler>/` chứa `fire_model.pkl`, `fire_scaler.pkl`, `train_meta.json`, `metrics.json` và các biểu đồ PR/ROC/CM. |
| `train_improve.py` | Phiên bản thay thế của `train.py` với thuật toán chọn threshold cân bằng recall giữa hai lớp (fire/not fire) khi vẫn thỏa điều kiện precision tối thiểu. Hữu ích khi muốn giảm bias về một lớp. | Quy trình sử dụng giống `train.py`: điều chỉnh `CONFIG` rồi chạy `python train_improve.py`. Các artifact cũng nằm trong `logs/<model>_<scaler>/`. |
| `train_dl.py` | Pipeline huấn luyện mô hình sâu (MLP, LSTM) cho dữ liệu bảng/tuyến tính, sử dụng PyTorch, early stopping và chuẩn hóa đầu vào. | Chuẩn hóa cấu hình tại `BASE_CONFIG` (đặc biệt `csv_path` chỉ tới file có cột `date` để tận dụng chuỗi thời gian). Chạy `python train_dl.py`; kết quả lưu trong `logs_dl/<model>_<scaler>/` gồm `model.pt`, `scaler.pkl`, `meta.json`, các biểu đồ và log đào tạo. |
| `app.py` | Ứng dụng Streamlit để tải model + scaler đã huấn luyện, dự đoán đơn lẻ hoặc theo lô từ CSV, đánh giá lại theo nhãn ground-truth (nếu có) và tải kết quả. | Cài đặt phụ thuộc `streamlit`, `scikit-learn`, `pandas`, `matplotlib`, `joblib`. Chạy `streamlit run app.py`, chọn thư mục log (mặc định `logs_dl`), model run và nhập thông số hoặc upload CSV. |
| `example_dataset.ipynb` | Notebook minh họa cách khám phá/tổng hợp dataset, dùng để thử nghiệm trực quan từng bước xử lý dữ liệu. | Mở notebook trong Jupyter hoặc VS Code, cập nhật đường dẫn dữ liệu theo máy cục bộ và chạy tuần tự các ô. |
| `test.ipynb` | Notebook để thử nghiệm nhanh một mô hình/chiến lược huấn luyện cụ thể (ví dụ: kiểm tra dữ liệu, chạy thủ công một mô hình nhỏ). | Dùng để ghi chú hoặc kiểm thử ad-hoc; không ảnh hưởng pipeline chính. |
| `data/` | Chứa dữ liệu gốc và dữ liệu đã xử lý (ERA5, MODIS, CSV sạch). | Tuân thủ cấu trúc `data/data/{era5,nasa,clean}` mà các script mong đợi. |
| `logs/`, `logs_dl/` | Nơi lưu lại artifact sau khi huấn luyện (model, scaler, metadata, biểu đồ, log). | Giữ nguyên cấu trúc vì `app.py` tìm trực tiếp trong đây. Có thể sao lưu/đổi tên các thư mục con để lưu lại nhiều thí nghiệm. |

## Quy trình gợi ý
1. **Chuẩn bị dữ liệu** – đặt file ERA5 & MODIS đúng vị trí rồi chạy `process_data.py` để tạo bộ dữ liệu `clean_*.csv`.
2. **Huấn luyện mô hình cổ điển** – chỉnh `CONFIG` trong `train.py` hoặc `train_improve.py` và chạy script tương ứng.
3. **Huấn luyện mô hình sâu (tùy chọn)** – dùng `train_dl.py` nếu muốn thử MLP/LSTM, đặc biệt với dữ liệu có chuỗi thời gian.
4. **Triển khai suy luận** – khởi chạy `app.py` bằng Streamlit, chọn thư mục chứa run mong muốn (`logs/` hoặc `logs_dl/`) và thực hiện dự đoán đơn/CSV.
