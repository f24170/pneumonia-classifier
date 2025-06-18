# Pneumonia Chest X-ray Classifier

本專案是一個使用 CNN 模型（以 PyTorch 架構）訓練的肺炎分類系統，可辨識胸腔 X 光影像是否為 **PNEUMONIA（肺炎）** 或 **NORMAL（正常）**。

## 功能特色
- CNN 模型使用 PyTorch 自建架構
- 單張或多張圖片預測（Gradio UI 支援拖曳上傳）
- GPU 加速支援（NVIDIA RTX）
- 支援模型訓練、儲存、準確率繪圖、預測輸出

## 快速開始 Demo

1️⃣ 安裝依賴套件
```bash
pip install -r requirements.txt
```

2️⃣ 下載資料集（從 Kaggle）
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d .
```

3️⃣ 訓練模型（含儲存最佳模型）
```bash
python train.py
```

4️⃣ 單張圖片推論
```bash
python predict.py chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg
```

5️⃣ 啟動 Gradio 圖形介面
```bash
python gradio_app.py
```

> 預設會開在 http://127.0.0.1:7860，可拖曳圖片即時預測

## 專案結構
```
.
├── model.py              # 模型定義
├── train.py              # 模型訓練主程式
├── predict.py            # 單張影像推論
├── gradio_app.py         # Gradio 圖形介面
├── utils.py              # 輔助工具函式（顯示圖片、繪圖）
├── saved_models/         # 儲存最佳模型
├── requirements.txt      # 所需套件
└── chest_xray/           # 資料集目錄（train/test/val）
```

## 資料集來源
- [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

此專案可作為 AI 醫療影像入門作品集，歡迎參考與改進 。
