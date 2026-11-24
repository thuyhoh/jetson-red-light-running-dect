import os
import cv2
import numpy as np
from ultralytics import YOLO


def is_red_from_traffic_light_box(frame, box_xyxy, red_ratio_threshold=0.08):
  h, w = frame.shape[:2]
  x1, y1, x2, y2 = map(int, box_xyxy)

  # Kẹp tọa độ trong biên khung hình
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(w, x2), min(h, y2)

  light_crop = frame[y1:y2, x1:x2]
  if light_crop.size == 0:
    return False, 0.0

  # Làm mượt nhẹ để giảm nhiễu hạt
  blurred = cv2.GaussianBlur(light_crop, (5, 5), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

  # Hai dải màu đỏ trên thang Hue [0..179] của OpenCV
  mask1 = cv2.inRange(
      hsv,
      np.array([0, 80, 80], dtype=np.uint8),
      np.array([10, 255, 255], dtype=np.uint8),
  )
  mask2 = cv2.inRange(
      hsv,
      np.array([170, 80, 80], dtype=np.uint8),
      np.array([180, 255, 255], dtype=np.uint8),
  )
  red_mask = cv2.bitwise_or(mask1, mask2)

  # Lọc nhiễu nhỏ
  kernel = np.ones((3, 3), np.uint8)
  red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

  red_pixels = cv2.countNonZero(red_mask)
  total_pixels = light_crop.shape[0] * light_crop.shape[1]
  red_ratio = red_pixels / max(1, total_pixels)

  return red_ratio >= red_ratio_threshold, red_ratio


output_folder = "frames_output"
os.makedirs(output_folder, exist_ok=True)

video_path = "/home/thuy/embedded/embedded_linux/jetson/red_light_running_dect/video02/1/7/16h.5.9.22.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  print("Không thể kết nối với video/camera!")
  exit()

model = YOLO("runs/detect/train-5/weights/best.pt")

roi_points = np.array(
    [(419, 206), (544, 209), (600, 326), (357, 317)],
    dtype=np.int32,
)
VEHICLE_CLASSES = [0, 5]
TRAFFIC_LIGHT_CLASS_ID = 1

STATE_COLORS = {
    "RED": (0, 0, 255),      # Đỏ
    "YELLOW": (0, 255, 255),  # Vàng
    "GREEN": (0, 255, 0),     # Xanh lá
}

confirmed_red = False
violated_ids = set()
frame_count = 0
current_light_state = "GREEN"
prev_inside_roi = {}

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = os.path.join(output_folder, "output_result.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
  ret, frame = cap.read()
  if not ret:
    print("Hết video hoặc không thể nhận dữ liệu. Đang thoát...")
    break

  # Reset danh sách xe cho từng frame mới
  vehicle_items = []
  frame_has_red = False

  target_classes = VEHICLE_CLASSES + [TRAFFIC_LIGHT_CLASS_ID]
  results = model.track(frame, persist=True, classes=target_classes, verbose=False)

  if results and results[0].boxes is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls_ids = results[0].boxes.cls.int().cpu().tolist()
    track_ids = (
        results[0].boxes.id.int().cpu().tolist()
        if results[0].boxes.id is not None
        else [-1] * len(boxes)
    )

    for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
      if cls_id == TRAFFIC_LIGHT_CLASS_ID:
        x1, y1, x2, y2 = map(int, box)
        is_red, ratio = is_red_from_traffic_light_box(
            frame, box, red_ratio_threshold=0.02
        )
        if is_red:
          frame_has_red = True

        light_color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), light_color, 2)
        cv2.putText(
            frame,
            f"Red Ratio: {ratio:.2f}",
            (max(5, x1 - 100), max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            light_color,
            2,
        )

      elif cls_id in VEHICLE_CLASSES:
        vehicle_items.append((box, cls_id, track_id))

  # Đồng bộ trạng thái đèn đỏ và màu ROI
  confirmed_red = frame_has_red
  current_light_state = "RED" if confirmed_red else "GREEN"
  roi_color = STATE_COLORS.get(current_light_state, (0, 255, 0))

  overlay = frame.copy()
  cv2.fillPoly(overlay, [roi_points], color=roi_color)
  display = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
  cv2.polylines(
      display, [roi_points], isClosed=True, color=roi_color, thickness=2
  )

  for box, cls_id, track_id in vehicle_items:
    x1, y1, x2, y2 = map(int, box)
    label_name = model.names[cls_id]

    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    inside_roi = (
        cv2.pointPolygonTest(roi_points, (foot_x, foot_y), measureDist=False) >= 0
    )

    if track_id != -1:
      was_inside = prev_inside_roi.get(track_id, False)
      if inside_roi and confirmed_red and (not was_inside or track_id in violated_ids):
        if track_id not in violated_ids:
          violated_ids.add(track_id)
          ev_path = os.path.join(
              output_folder, f"violation_id{track_id}_f{frame_count:06d}.jpg"
          )
          cv2.imwrite(ev_path, display)

      prev_inside_roi[track_id] = inside_roi

    # Luôn gán màu và nhãn kể cả khi track_id == -1
    if track_id != -1 and track_id in violated_ids:
      veh_color = (0, 0, 255)
      status_label = f"ID:{track_id} {label_name} [VUOT DEN DO]"
    elif inside_roi:
      id_text = f"ID:{track_id} " if track_id != -1 else ""
      veh_color = (0, 255, 255)
      status_label = f"{id_text}{label_name} [IN ROI]"
    else:
      id_text = f"ID:{track_id} " if track_id != -1 else ""
      veh_color = (255, 200, 0)
      status_label = f"{id_text}{label_name}"

    cv2.putText(
        display,
        status_label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        veh_color,
        2,
    )
    cv2.rectangle(display, (x1, y1), (x2, y2), veh_color, 2)
    cv2.circle(display, (foot_x, foot_y), 4, veh_color, -1)

  cv2.putText(
      display,
      f"Light: {current_light_state} | Violations: {len(violated_ids)}",
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      roi_color,
      2,
  )

  out.write(display)
  cv2.imshow("Camera Stream", display)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break


cap.release()
cv2.destroyAllWindows()