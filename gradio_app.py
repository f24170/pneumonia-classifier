import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

from model import PneumoniaCNN

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=device))
model.eval()

# 類別名稱
class_names = ["NORMAL", "PNEUMONIA"]

# 圖片預處理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 預測函數
def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, 0)
        return {class_names[0]: float(probs[0]), class_names[1]: float(probs[1])}

# 建立 Gradio 介面
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🫁 肺炎影像分類系統",
    description="上傳或拖曳胸腔 X 光影像，模型將判斷是否有肺炎跡象（PNEUMONIA / NORMAL）",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
