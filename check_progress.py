import os
import numpy as np

output_dir = "datasets/amazon_reviews23/processed"
total_items = 49930

print(f"CLIP特征生成进度监控 (目标: {total_items} 商品)")
print("=" * 60)

images_dir = "datasets/amazon_reviews23/images"
img_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])
print(f"图片总数: {img_count}")

features_path = os.path.join(output_dir, "multimodal_features.npy")
if os.path.exists(features_path):
    data = np.load(features_path, allow_pickle=True).item()
    done = len(data)
    pct = done / total_items * 100
    print(f"已生成特征: {done}/{total_items} ({pct:.1f}%)")
    print(f"特征文件大小: {os.path.getsize(features_path) / 1024 / 1024:.1f} MB")
else:
    print("特征文件尚未生成（处理完成后或每5000个checkpoint时写入）")

print()
print("实时进度请查看IDE底部终端面板的tqdm进度条")
print("格式: XX%|████| 已完成/总数 [已用时间<剩余时间, 速度it/s]")
