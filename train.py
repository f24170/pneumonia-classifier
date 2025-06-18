import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import PneumoniaResNet18
from utils import show_images, plot_accuracy

# 檢查裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")
if torch.cuda.is_available():
    print(f"GPU 型號：{torch.cuda.get_device_name(0)}")

# 資料預處理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 讀取資料集
train_dataset = ImageFolder(root="chest_xray/train", transform=transform)
test_dataset = ImageFolder(root="chest_xray/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型、損失函數、優化器
model = PneumoniaResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練參數
num_epochs = 10
best_acc = 0.0
train_acc_list = []

os.makedirs("saved_models", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 評估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    train_acc_list.append(acc)
    print(f"📘 Epoch {epoch+1}/{num_epochs} - Accuracy: {acc:.2f}%")

    # 儲存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "saved_models/best_model.pth")
        print(f"新最佳模型已儲存（準確率 {acc:.2f}%）")
    else:
        print(f"本輪未超越最佳準確率（{best_acc:.2f}%），未儲存")

# 畫圖
plot_accuracy(train_acc_list)

# 最後提示
print(f"訓練完成，最佳準確率：{best_acc:.2f}%")
print("模型儲存路徑：saved_models/best_model.pth")
