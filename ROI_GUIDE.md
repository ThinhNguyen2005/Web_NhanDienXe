# Hướng dẫn sử dụng hệ thống ROI mới

## Tổng quan
Hệ thống đã được cập nhật để sử dụng **Region of Interest (ROI)** thay vì phương pháp stop line cũ. Điều này cho phép phát hiện vi phạm chính xác hơn dựa trên các vùng đa giác tùy chỉnh.

## Các thay đổi chính

### 1. Từ Stop Line sang ROI
- **Trước**: Hệ thống sử dụng một đường thẳng ngang để xác định vi phạm
- **Sau**: Hệ thống sử dụng hai vùng đa giác:
  - **Vùng chờ** (màu vàng): Khu vực xe được phép dừng chờ đèn
  - **Vùng vi phạm** (màu đỏ): Khu vực xe không được phép vào khi đèn đỏ

### 2. Cách hoạt động mới
1. **Khi đèn đỏ**: Xe trong vùng vi phạm sẽ bị ghi nhận vi phạm
2. **Vùng chờ**: Xe trong vùng này không bị tính vi phạm (ngay cả khi đèn đỏ)
3. **Khi đèn xanh/vàng**: Không có vi phạm được ghi nhận

## Cách sử dụng

### Bước 1: Thiết lập ROI
1. Truy cập: **Thiết lập ROI** từ menu navigation
2. Chọn video mẫu hoặc tải lên video mới
3. Nhập **Camera ID** (ví dụ: "camera_1", "default")
4. Có hai cách thiết lập:

#### Cách 1: Tự động phát hiện
- Nhấn nút **"Tự động phát hiện vùng ROI"**
- Hệ thống sẽ tự động tạo vùng dựa trên đèn giao thông

#### Cách 2: Vẽ thủ công
- Nhấn **"Vẽ vùng chờ"** → click chuột để tạo đa giác màu vàng
- Nhấn **"Vẽ vùng vi phạm"** → click chuột để tạo đa giác màu đỏ
- Nhấn **"Lưu thiết lập"**

### Bước 2: Xử lý video
1. **Upload video** như bình thường
2. Hệ thống sẽ tự động:
   - Load ROI configuration cho camera tương ứng (dựa trên tên file)
   - Áp dụng ROI đã cấu hình cho toàn bộ video
   - Nếu chưa có ROI, sẽ thử auto-detect

### Bước 3: Xem kết quả
- Video được xử lý sẽ hiển thị vùng ROI được vẽ lên
- Vi phạm được đánh dấu bằng khung đỏ xung quanh xe
- Thông tin biển số sẽ được trích xuất và lưu

## Ưu điểm của hệ thống ROI

### 1. Chính xác hơn
- Phù hợp với hình dạng thực tế của giao lộ
- Có thể thiết lập theo góc camera và vị trí cụ thể
- Giảm thiểu false positive

### 2. Linh hoạt
- Có thể cấu hình khác nhau cho từng camera
- Dễ dàng điều chỉnh khi thay đổi góc camera
- Hỗ trợ giao lộ phức tạp

### 3. Trực quan
- Vùng được hiển thị rõ ràng trên video
- Dễ hiểu và kiểm tra kết quả

## Cấu trúc file

### ROI Configuration
- Lưu tại: `config/rois/{camera_id}.json`
- Format:
```json
{
  "waiting_zone": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "violation_zone": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
}
```

### Camera ID Mapping
- Video file: `traffic_video.mp4` → Camera ID: `traffic_video`
- Có thể override bằng cách set camera ID khác trong ROI config

## Troubleshooting

### Vấn đề: Không phát hiện vi phạm
**Nguyên nhân có thể:**
1. Chưa cấu hình ROI cho camera
2. Vùng vi phạm quá nhỏ hoặc sai vị trí
3. Đèn giao thông không được phát hiện

**Giải pháp:**
1. Kiểm tra ROI config tại: `config/rois/{camera_id}.json`
2. Cấu hình lại ROI qua trang "Thiết lập ROI"
3. Test với video có đèn giao thông rõ ràng

### Vấn đề: Quá nhiều false positive
**Giải pháp:**
1. Thu nhỏ vùng vi phạm
2. Mở rộng vùng chờ
3. Điều chỉnh vị trí vùng cho phù hợp

## API Endpoints mới

- `GET /roi_config` - Trang cấu hình ROI
- `POST /api/save_roi` - Lưu cấu hình ROI
- `POST /api/auto_detect_roi` - Tự động phát hiện ROI
- `GET /api/load_roi/{camera_id}` - Load ROI đã lưu

## Log và Debug

Khi xử lý video, check log để xem:
```
INFO - Loaded ROI for camera {camera_id}:
INFO -   - Waiting zone: X points
INFO -   - Violation zone: Y points
```

Nếu thấy "Could not auto-detect ROI", cần cấu hình thủ công.