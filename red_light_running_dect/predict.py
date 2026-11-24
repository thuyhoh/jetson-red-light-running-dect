from ultralytics import YOLO

if __name__ == "__main__":
  # 1. Load trọng số tốt nhất sau khi train xong (thay đường dẫn train cho đúng thực tế)
  model = YOLO("runs/detect/train-5/weights/best.pt")

  # 2. Chạy dự đoán trên ảnh, video hoặc webcam
  # source có thể là: đường dẫn file ảnh, đường dẫn video (như 'video02'), hoặc 0 (webcam)
  results = model.predict(
      source="/home/thuy/embedded/embedded_linux/jetson/red_light_running_dect/video02/1/1/16h.5.9.22.mp4",  # Hoặc đường dẫn tới một video/ảnh cụ thể
      conf=0.25,  # Ngưỡng tin cậy (confidence threshold)
      show=True,  # Hiển thị cửa sổ popup kết quả trực quan
      save=True,  # Lưu kết quả ra thư mục runs/predict/
  )

  print("Đã chạy test hoàn tất! Kết quả được lưu trong thư mục runs/predict/")