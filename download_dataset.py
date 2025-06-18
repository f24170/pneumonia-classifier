import os
import zipfile

# Kaggle 認證（透過 GitHub Secrets）
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

# 自動下載與解壓
os.system('kaggle datasets download -d paultimothymooney/chest-xray-pneumonia')
with zipfile.ZipFile("chest-xray-pneumonia.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
