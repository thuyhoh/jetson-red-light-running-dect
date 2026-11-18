from ultralytics import YOLO

if __name__ == "__main__":
  # Tải mô hình pre-trained (ví dụ: yolov8n.pt hoặc yolov8s.pt)
  model = YOLO("yolov8n.pt")

  # Huấn luyện mô hình
  results = model.train(
      data="data.yaml",  # Đường dẫn tới file yaml vừa tạo
      epochs=50,  # Số vòng train (tùy chỉnh theo dung lượng dataset)
      imgsz=640,  # Kích thước ảnh đầu vào (có thể tăng lên 1280 nếu đèn giao thông quá nhỏ)
      batch=16,  # Số lượng mẫu mỗi batch (tùy thuộc VRAM GPU)
      device=0,  # Sử dụng GPU (0) hoặc CPU ('cpu')
  )