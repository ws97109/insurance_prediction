# 🚀 DQN + XGBoost 保險預測系統 - 啟動指南

## ✅ 系統已整理完成

所有舊檔案已刪除，系統現在完全以 **DQN + XGBoost** 為核心！

## 📁 當前檔案結構

```
insurance_perdiction/
├── dqn_xgboost_model.py       # DQN + XGBoost 核心模型
├── train_model.py              # 模型訓練腳本
├── app.py                      # Flask Web 應用（Port 8080）
├── trained_model.pkl           # 已訓練好的模型
├── dqn_xgboost_analysis.png   # 訓練分析圖表
├── templates/
│   └── index.html              # Web 前端介面
├── requirements.txt            # 依賴套件
├── README.md                   # 完整使用文檔
└── START_GUIDE.md             # 本指南
```

## 🎯 快速啟動（3步驟）

### 步驟 1: 確認環境

```bash
# 確認已安裝依賴
pip install -r requirements.txt
```

### 步驟 2: 啟動系統

```bash
# 啟動 DQN + XGBoost 預測系統
python3 app.py
```

### 步驟 3: 開始使用

在瀏覽器中訪問:
```
http://localhost:8080
```

**注意**: 系統運行在 **Port 8080**（因為 5000 被 AirPlay 佔用）

## 🎨 功能介面

### 1️⃣ 單筆預測
- 輸入單個客戶的 85 個特徵
- DQN 自動提取 32 個深度特徵
- XGBoost 使用 117 個特徵進行預測
- 顯示購買機率和信心度

### 2️⃣ 批量預測
- 上傳 CSV 或 TXT 檔案
- 批量處理多個客戶
- 查看統計摘要
- 下載完整預測結果

### 3️⃣ JSON API
- RESTful API 端點
- 支援程式化調用
- 完整的 JSON 回應

### 4️⃣ API 文檔
- 查看所有 API 端點
- 模型架構說明
- 使用範例

## 🔬 模型資訊

- **架構**: Deep Q-Network (3層) + XGBoost (200棵樹)
- **特徵**: 85 原始特徵 + 32 DQN 深度特徵 = 117 總特徵
- **測試準確率**: 91.59%
- **ROC-AUC**: 56.09%
- **訓練時間**: ~2-3 分鐘

## 📡 API 端點

### 基礎 URL
```
http://localhost:8080
```

### 主要端點

1. **模型資訊**
   ```bash
   curl http://localhost:8080/api/info
   ```

2. **單筆預測**
   ```bash
   curl -X POST http://localhost:8080/api/predict \
     -H "Content-Type: application/json" \
     -d @customer.json
   ```

3. **批量預測**
   ```bash
   curl -X POST http://localhost:8080/api/predict_batch \
     -F "file=@customers.csv"
   ```

4. **特徵重要性**
   ```bash
   curl http://localhost:8080/api/feature_importance?top_n=20
   ```

5. **下載結果**
   ```bash
   curl -X POST http://localhost:8080/api/predict_batch/download \
     -F "file=@customers.csv" \
     -o predictions.csv
   ```

6. **範例輸入**
   ```bash
   curl http://localhost:8080/api/example
   ```

## 🔄 重新訓練模型

如果需要重新訓練模型：

```bash
python3 train_model.py
```

訓練過程會經過：
1. **Phase 1**: DQN 強化學習（50 episodes）
2. **Phase 2**: DQN 特徵提取（32 維）
3. **Phase 3**: XGBoost 訓練（117 維）

訓練完成後會：
- 保存模型到 `trained_model.pkl`
- 生成分析圖表 `dqn_xgboost_analysis.png`
- 顯示特徵重要性

## 💡 快速測試

### 測試 API
```bash
# 測試模型資訊
curl http://localhost:8080/api/info | python3 -m json.tool

# 獲取範例輸入
curl http://localhost:8080/api/example | python3 -m json.tool
```

### Python 測試腳本
```python
import requests

# 單筆預測
response = requests.get('http://localhost:8080/api/example')
example_data = response.json()['example']

result = requests.post(
    'http://localhost:8080/api/predict',
    json=example_data
)

print("預測結果:", result.json())
```

## 🎯 使用案例

### 案例 1: 行銷活動
篩選高潛力客戶進行精準行銷：
```python
import pandas as pd
import requests

df = pd.read_csv('customers.csv')
data = df.to_dict('records')

response = requests.post(
    'http://localhost:8080/api/predict_batch',
    json=data
)

results = response.json()
high_potential = [
    p for p in results['predictions']
    if p['probability_will_buy'] > 0.3
]

print(f"高潛力客戶: {len(high_potential)}")
```

### 案例 2: 即時評估
客服系統即時評估客戶購買意願：
```python
def evaluate_customer(customer_data):
    response = requests.post(
        'http://localhost:8080/api/predict',
        json=customer_data
    )
    result = response.json()

    probability = result['probability']['will_buy']

    if probability > 0.3:
        return "高度推薦"
    elif probability > 0.15:
        return "可以推薦"
    else:
        return "暫不推薦"
```

## 🔧 問題排解

### Port 被佔用
如果 8080 也被佔用，修改 `app.py` 最後一行：
```python
app.run(debug=True, host='0.0.0.0', port=9000)  # 改成其他 port
```

### 模型載入失敗
確認 `trained_model.pkl` 存在：
```bash
ls -lh trained_model.pkl
```

如果不存在，執行：
```bash
python3 train_model.py
```

### 依賴套件問題
重新安裝所有依賴：
```bash
pip install -r requirements.txt --upgrade
```

## 📊 查看訓練結果

訓練完成後會生成視覺化圖表：
```bash
open dqn_xgboost_analysis.png  # macOS
# 或
xdg-open dqn_xgboost_analysis.png  # Linux
```

圖表包含：
- 混淆矩陣
- ROC 曲線
- 特徵重要性（含 DQN 特徵）
- 預測機率分布
- DQN 訓練損失
- 特徵類型分布

## 🎓 進階配置

### 增加訓練輪數
編輯 `train_model.py`，找到：
```python
model.train(..., dqn_episodes=50)
```
改為：
```python
model.train(..., dqn_episodes=100)  # 更好的性能
```

### 調整網路架構
編輯 `dqn_xgboost_model.py`，修改：
```python
DQN(input_dim, hidden_dims=[512, 256, 128])  # 更深的網路
```

### 使用 GPU 加速
系統會自動檢測 GPU，如果有 CUDA：
```python
# 訓練時間從 3 分鐘降至 30 秒
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 📚 完整文檔

詳細資訊請查看：
- **[README.md](README.md)** - 完整技術文檔
- **[dqn_xgboost_model.py](dqn_xgboost_model.py)** - 模型源碼

## ✨ 系統特色

✅ **已刪除的舊檔案**:
- ❌ `train_dqn_xgboost.py` → ✅ 改名為 `train_model.py`
- ❌ `app_dqn_xgboost.py` → ✅ 改名為 `app.py`
- ❌ `index_dqn.html` → ✅ 改名為 `index.html`
- ❌ `trained_dqn_xgboost_model.pkl` → ✅ 改名為 `trained_model.pkl`
- ❌ `requirements_dqn.txt` → ✅ 改名為 `requirements.txt`
- ❌ 舊的 `train_model.py` (Gradient Boosting)
- ❌ 舊的 `app.py` (非 DQN 版本)
- ❌ 舊的 `batch_predict.py`
- ❌ 舊的 `README.md` 和 `USAGE_GUIDE.md`

✅ **現在的系統**:
- ✅ 完全基於 DQN + XGBoost
- ✅ 統一的檔案命名
- ✅ 簡潔的目錄結構
- ✅ 完整的文檔

## 🎉 開始使用

```bash
# 就這麼簡單！
python3 app.py
```

然後訪問: **http://localhost:8080**

---

**🤖 享受 DQN + XGBoost 帶來的智能預測！**
