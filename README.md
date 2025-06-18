
#  Pneumonia Classifier

本專案為使用 CNN 模型訓練的**肺炎影像分類器**，能夠辨識肺部 X 光影像為「正常」或「肺炎」。提供 Gradio 網頁介面供使用者上傳圖片並即時預測。

---

##  專案結構

```
pneumonia-classifier/
├── chest_xray/                # 解壓後的影像資料集（由程式自動下載）
├── download_dataset.py       # 用於下載並解壓 Kaggle 資料集
├── train.py                  # 模型訓練程式
├── predict.py                # 單張預測測試
├── model.py                  # 模型架構定義
├── utils.py                  # 資料視覺化與輔助函式
├── gradio_app.py             # Gradio 圖形介面
├── requirements.txt          # Python 相依套件
├── README.md                 # 本說明文件
└── .github/workflows/
    └── python-test.yml       # GitHub Actions 自動測試與資料集下載
```

---

##  安裝依賴

請先安裝 Python 3.10 或以上版本，並安裝 PyTorch 對應 GPU（或 CPU）版本。

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

✅ 若你要使用 GPU，請修改 `requirements.txt` 為：

```
torch==2.2.2+cu118
torchvision==0.17.2+cu118
--extra-index-url https://download.pytorch.org/whl/cu118
```

---

##  資料集下載（Kaggle API）

1. 至 https://www.kaggle.com/account 設定並下載 `kaggle.json`
2. 將 API 金鑰設為環境變數或 `.env` 檔案（或用 GitHub Secret）
3. 執行以下程式自動下載與解壓：

```bash
python download_dataset.py
```

---

##  模型訓練

```bash
python train.py
```

預設會使用 `chest_xray/train` 與 `val` 資料夾訓練模型，並將模型儲存至 `saved_models/`。

---

##  單張預測

```bash
python predict.py
```

可讀取圖片進行單張測試，並輸出模型預測與信心值。

---

##  啟動 Gradio Web 介面

```bash
python gradio_app.py
```

啟動後會打開本機端的網頁（預設為 http://127.0.0.1:7860）。

---

##  GitHub Actions 測試（CI）

本專案支援 GitHub Actions，當有 Push 時會自動：

1. 安裝依賴
2. 下載並解壓 Kaggle 資料集
3. 測試 `train.py` 和 `predict.py` 是否可執行

請確認 `.github/workflows/python-test.yml` 正確設置並且已設定 `KAGGLE_USERNAME` 與 `KAGGLE_KEY` 為 Secret。

---

## `.gitignore` 注意事項

以下項目已忽略，不會被加入 Git：

- `chest_xray/`、`saved_models/`、`.zip`
- `.env`、虛擬環境 `venv/`
- `.ipynb_checkpoints/`、`.DS_Store`

---

##  參考資料

- [Kaggle Dataset - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://www.gradio.app/)

---

##  開發者

余峻廷 | 2025  
目前正在學習 AI / 深度學習技術，打造職業轉職作品集
