import os
import cv2

video_path = "/home/thuy/embedded/embedded_linux/jetson/red_light_running_dect/video02/1/7/16h.25.9.22.mp4"
save_dir = "extracted_frames"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
  print("Không thể mở video!")
  exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
  fps = 25.0

# Ví dụ: lấy 1 frame mỗi 5 frame (đặt = 1 nếu muốn lấy toàn bộ)
frame_step = 5

frame_idx = 0
saved_count = 0

while True:
  ret, frame = cap.read()
  if not ret:
    break

  if frame_idx % frame_step == 0:
    out_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, frame)
    saved_count += 1

  frame_idx += 1

cap.release()
print(f"Đã đọc {frame_idx} frame, lưu {saved_count} frame vào '{save_dir}'.")