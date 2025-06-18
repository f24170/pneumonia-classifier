import torch
from torchvision import transforms
from utils import print_prediction_with_confidence
from PIL import Image
import sys
import os

from model import PneumoniaCNN

# 檢查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")
if torch.cuda.is_available():
    print(f"GPU 型號：{torch.cuda.get_device_name(0)}")
    
# 載入模型
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=device))
model.eval()

# 影像預處理流程（和訓練時一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 類別名稱（根據訓練集資料夾順序）
class_names = ["NORMAL", "PNEUMONIA"]

# 預測函式
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ 找不到圖片：{image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        print_prediction_with_confidence(outputs, class_names)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        print(f"✅ 預測結果：{label}")

# 主程式
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請提供圖片路徑：")
        print("使用方式：python predict.py 圖片路徑")
    else:
        predict_image(sys.argv[1])
