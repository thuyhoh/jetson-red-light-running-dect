import json
import cv2
import numpy as np

img = cv2.imread("/home/thuy/embedded/embedded_linux/jetson/red_light_running_dect/extracted_frames/frame_000000.jpg")
if img is None:
  print("Không đọc được file ảnh.")
  exit()

points = []
window_name = "Ve Da Giac (Click chon diem, S: luu, R: reset, Q: thoat)"


def mouse_handler(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    points.append((x, y))


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_handler)

while True:
  canvas = img.copy()

  for idx, pt in enumerate(points):
    cv2.circle(canvas, pt, 5, (0, 255, 255), -1)
    cv2.putText(
        canvas,
        str(idx + 1),
        (pt[0] + 6, pt[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

  if len(points) > 1:
    cv2.polylines(
        canvas, [np.array(points, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2
    )

  if len(points) >= 3:
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [np.array(points, np.int32)], (0, 255, 0))
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    cv2.polylines(
        canvas, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2
    )

  cv2.imshow(window_name, canvas)
  key = cv2.waitKey(20) & 0xFF

  if key == ord("r"):
    points.clear()
  elif key == ord("s") and len(points) >= 3:
    with open("polygon_roi.json", "w", encoding="utf-8") as f:
      json.dump({"points": points}, f, indent=2)
    print("Đã lưu đa giác vào polygon_roi.json:", points)
  elif key in (ord("q"), 27):
    break

cv2.destroyAllWindows()