import os
import random
import shutil

# Thư mục nguồn chứa toàn bộ ảnh đã cắt
source_dir = "dataset/all_captured_images"
# Thư mục đích để lưu 300 ảnh ngẫu nhiên
target_dir = "dataset/random_300_images"
num_images = 300

os.makedirs(target_dir, exist_ok=True)

# Lấy danh sách tất cả file ảnh hợp lệ trong thư mục nguồn
valid_extensions = (".jpg", ".jpeg", ".png")
all_files = [
    f for f in os.listdir(source_dir) 
    if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(valid_extensions)
]

# Kiểm tra nếu tổng số ảnh ít hơn 300 thì lấy toàn bộ
sample_size = min(num_images, len(all_files))
selected_files = random.sample(all_files, sample_size)

# Copy các ảnh được chọn sang thư mục mới
for file_name in selected_files:
    src_path = os.path.join(source_dir, file_name)
    dst_path = os.path.join(target_dir, file_name)
    shutil.copy2(src_path, dst_path)  # Đổi thành shutil.move(src_path, dst_path) nếu muốn cắt hẳn sang

print(f"Hoàn tất! Đã copy ngẫu nhiên {sample_size} ảnh vào thư mục: {target_dir}")