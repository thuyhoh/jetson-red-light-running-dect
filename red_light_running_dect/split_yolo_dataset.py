import os
import random
import shutil
from pathlib import Path


def split_dataset(
    img_dir, label_dir, output_dir, train_ratio=0.8, seed=42
):
  random.seed(seed)

  # Tạo cấu trúc thư mục đầu ra
  for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

  # Lấy danh sách tất cả các file ảnh (hỗ trợ .jpg, .png, .jpeg)
  valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
  image_files = [
      f
      for f in os.listdir(img_dir)
      if f.lower().endswith(valid_extensions)
  ]

  # Lọc ra những ảnh thực sự có file nhãn (.txt) tương ứng
  valid_pairs = []
  for img_name in image_files:
    base_name, _ = os.path.splitext(img_name)
    label_name = base_name + ".txt"
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, label_name)

    if os.path.exists(label_path):
      valid_pairs.append((img_path, label_path))
    else:
      print(f"Cảnh báo: Không tìm thấy nhãn cho ảnh {img_name}, bỏ qua.")

  # Xáo trộn ngẫu nhiên danh sách file
  random.shuffle(valid_pairs)

  # Tính số lượng cho tập train
  split_index = int(len(valid_pairs) * train_ratio)
  train_pairs = valid_pairs[:split_index]
  val_pairs = valid_pairs[split_index:]

  # Hàm copy file vào thư mục tương ứng
  def copy_files(pairs, split):
    for img_p, lbl_p in pairs:
      # Copy ảnh
      shutil.copy(img_p, os.path.join(output_dir, "images", split, os.path.basename(img_p)))
      # Copy nhãn
      shutil.copy(lbl_p, os.path.join(output_dir, "labels", split, os.path.basename(lbl_p)))

  # Thực hiện copy
  copy_files(train_pairs, "train")
  copy_files(val_pairs, "val")

  print("Hoàn tất chia dữ liệu!")
  print(f"- Tổng số mẫu hợp lệ: {len(valid_pairs)}")
  print(f"- Tập Train: {len(train_pairs)} mẫu")
  print(f"- Tập Val: {len(val_pairs)} mẫu")


# === CẤU HÌNH ĐƯỜNG DẪN THỰC TẾ CỦA BẠN ===
if __name__ == "__main__":
  # Thư mục chứa ảnh gốc và nhãn gốc của bạn
  IMAGE_DIR = "dataset/imgs"
  LABEL_DIR = "dataset/lb"

  # Thư mục đích sau khi chia xong (có thể để nguyên hoặc tạo mới)
  OUTPUT_DIR = "dataset_split"

  # Tỷ lệ train (80% train, 20% val)
  TRAIN_RATIO = 0.8

  split_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, train_ratio=TRAIN_RATIO)