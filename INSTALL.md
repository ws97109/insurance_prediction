# 🔧 DQN + XGBoost 系統安裝指南

## 📋 系統需求

- **Python**: 3.8 或以上（已測試 Python 3.12）
- **作業系統**: macOS, Linux, Windows
- **記憶體**: 建議 4GB 以上
- **硬碟空間**: 約 500MB（含依賴套件）

## 🚀 安裝步驟

### 方式一：自動安裝（推薦）

```bash
# 1. 進入專案目錄
cd /Users/lishengfeng/Desktop/insurance_perdiction

# 2. 安裝所有依賴套件
pip install -r requirements.txt

# 3. 完成！啟動系統
python3 app.py
```

### 方式二：手動安裝

```bash
# 核心套件
pip install flask flask-cors

# 數據處理
pip install pandas numpy

# 機器學習
pip install scikit-learn xgboost

# 深度學習
pip install torch

# 視覺化
pip install matplotlib seaborn

# 其他工具
pip install gymnasium tensorboard
```

## ✅ 驗證安裝

### 檢查 Python 版本
```bash
python3 --version
# 應顯示: Python 3.8+ (例如 Python 3.12.2)
```

### 檢查套件安裝
```bash
python3 -c "import torch; import xgboost; import flask; print('✅ 所有套件已安裝')"
```

### 檢查模型檔案
```bash
ls -lh trained_model.pkl
# 應顯示: trained_model.pkl (約 8.2 MB)
```

## 🎯 快速測試

### 啟動系統
```bash
python3 app.py
```

應該看到：
```
Loading DQN + XGBoost hybrid model...
Model loaded successfully!

================================================================================
DQN + XGBoost Insurance Prediction API Server
================================================================================
Model: DQN + XGBoost Hybrid
Features: 85 original + 32 DQN features = 117 total
Test Accuracy: 0.9159
Test ROC-AUC: 0.5609

Starting server on http://localhost:8080
================================================================================
```

### 測試 API
打開新的終端視窗：
```bash
curl http://localhost:8080/api/info
```

應該返回 JSON 格式的模型資訊。

### 測試 Web 介面
在瀏覽器開啟：
```
http://localhost:8080
```

應該看到漂亮的 DQN + XGBoost 預測系統介面。

## 🔧 常見問題

### 問題 1: torch 安裝失敗

**錯誤訊息**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解決方案**:
```bash
# 使用最新版本
pip install torch --upgrade

# 或指定特定版本（根據 Python 版本）
pip install torch>=2.2.0
```

### 問題 2: Port 8080 被佔用

**錯誤訊息**:
```
Address already in use
```

**解決方案 1** - 停止佔用的程序:
```bash
# 查找佔用的程序
lsof -ti:8080

# 停止程序
kill -9 $(lsof -ti:8080)
```

**解決方案 2** - 使用其他 Port:
編輯 `app.py` 最後一行：
```python
app.run(debug=True, host='0.0.0.0', port=9000)  # 改成 9000
```

### 問題 3: 模型檔案不存在

**錯誤訊息**:
```
FileNotFoundError: trained_model.pkl
```

**解決方案**:
```bash
# 重新訓練模型（需要 2-3 分鐘）
python3 train_model.py
```

### 問題 4: numpy 版本衝突

**錯誤訊息**:
```
numpy version conflict
```

**解決方案**:
```bash
pip install numpy --upgrade
pip install pandas --upgrade
```

### 問題 5: macOS AirPlay 佔用 Port 5000

**解決方案**:
系統已經改用 Port 8080，不受影響。

如果需要使用 Port 5000：
1. 開啟「系統設定」
2. 進入「一般」→「AirDrop 與 Handoff」
3. 關閉「AirPlay 接收器」

## 📦 依賴套件清單

| 套件 | 版本 | 用途 |
|------|------|------|
| flask | ≥3.0.0 | Web 框架 |
| flask-cors | ≥4.0.0 | 跨域請求 |
| pandas | ≥2.0.0 | 數據處理 |
| numpy | ≥1.24.0 | 數值計算 |
| scikit-learn | ≥1.3.0 | 機器學習工具 |
| matplotlib | ≥3.7.0 | 視覺化 |
| seaborn | ≥0.12.0 | 統計視覺化 |
| xgboost | ≥2.0.0 | 梯度提升 |
| torch | ≥2.2.0 | 深度學習 |
| gymnasium | ≥0.29.0 | 強化學習環境 |
| tensorboard | ≥2.15.0 | 訓練監控 |

## 🐳 Docker 安裝（可選）

如果想使用 Docker：

### Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]
```

### 建立和執行
```bash
# 建立映像
docker build -t dqn-xgboost-insurance .

# 執行容器
docker run -p 8080:8080 dqn-xgboost-insurance
```

## 🌐 虛擬環境（推薦）

建議使用虛擬環境避免套件衝突：

### 使用 venv
```bash
# 創建虛擬環境
python3 -m venv venv

# 啟動虛擬環境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 啟動系統
python3 app.py

# 完成後退出虛擬環境
deactivate
```

### 使用 conda
```bash
# 創建環境
conda create -n dqn-xgboost python=3.12

# 啟動環境
conda activate dqn-xgboost

# 安裝依賴
pip install -r requirements.txt

# 啟動系統
python3 app.py

# 完成後退出
conda deactivate
```

## 💻 開發環境設定

### VS Code
推薦安裝擴充功能：
- Python
- Pylance
- Jupyter

### PyCharm
已包含所有需要的功能。

### Jupyter Notebook
如果要使用 Notebook：
```bash
pip install jupyter
jupyter notebook
```

## 🚀 生產環境部署

### 使用 Gunicorn
```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### 使用 Nginx（反向代理）
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 效能優化

### GPU 加速（可選）
如果有 NVIDIA GPU：
```bash
# 安裝 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

系統會自動檢測並使用 GPU。

### 記憶體優化
如果記憶體不足：
1. 減少 DQN episodes（在 `train_model.py` 中）
2. 減少 XGBoost 樹的數量
3. 使用批次處理而非一次載入全部資料

## ✅ 安裝完成檢查清單

- [ ] Python 3.8+ 已安裝
- [ ] 所有依賴套件已安裝
- [ ] `trained_model.pkl` 存在（8.2 MB）
- [ ] `python3 app.py` 可以成功啟動
- [ ] http://localhost:8080 可以訪問
- [ ] API 測試成功（`curl http://localhost:8080/api/info`）

## 📞 需要幫助？

如果遇到問題：
1. 查看 [START_GUIDE.md](START_GUIDE.md)
2. 查看 [README.md](README.md)
3. 檢查錯誤訊息
4. 開啟 GitHub Issue

---

**🎉 安裝完成，開始使用 DQN + XGBoost 系統！**

```bash
python3 app.py
```
