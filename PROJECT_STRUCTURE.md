# 專案結構說明

## 📁 檔案清單

```
insurance_perdiction/
│
├── 📄 README.md                          # 專案說明（快速開始）
├── 📄 RECOMMENDATION_GUIDE.md            # 完整使用指南
├── 📄 CROSS_SELL_SUMMARY.md              # 系統功能總結
├── 📄 PROJECT_STRUCTURE.md               # 本文件
│
├── 🐍 multi_insurance_model.py           # 多保險預測模型訓練腳本
├── 🐍 recommendation_app.py              # API 服務主程式
├── 🐍 test_recommendation.py             # 測試腳本
│
├── 💾 multi_insurance_model.pkl          # 訓練好的模型 (2.8 MB)
├── 📋 requirements.txt                   # Python 依賴套件
│
├── 📂 insurance+company+benchmark+coil+2000/  # 訓練資料集
│   ├── ticdata2000.txt                   # 訓練資料 (5,822 筆)
│   ├── dictionary.txt                    # 欄位說明
│   └── ...
│
└── 📂 ccpm/                              # 專案配置（可選）
```

## 📝 核心檔案說明

### Python 程式

#### 1. `multi_insurance_model.py` (14 KB)
**功能**: 訓練多標籤保險預測模型
- 載入訓練資料
- 訓練 DQN 特徵提取器
- 為 19 種保險建立 XGBoost 分類器
- 儲存訓練好的模型

**使用**:
```bash
python multi_insurance_model.py
```

**輸出**: `multi_insurance_model.pkl`

---

#### 2. `recommendation_app.py` (11 KB)
**功能**: RESTful API 服務
- 載入訓練好的模型
- 提供推薦 API 端點
- 支援單一和批量推薦
- 自動過濾已購買保險

**使用**:
```bash
python recommendation_app.py
# 啟動在 http://localhost:5000
```

**API 端點**:
- `GET /` - API 資訊
- `POST /api/recommend` - 單一客戶推薦
- `POST /api/recommend_batch` - 批量推薦
- `GET /api/products` - 保險產品列表
- `GET /api/example` - 範例資料
- `GET /api/health` - 健康檢查

---

#### 3. `test_recommendation.py` (5.3 KB)
**功能**: 測試推薦系統
- 建立測試客戶資料
- 呼叫推薦功能
- 展示輸出格式

**使用**:
```bash
python test_recommendation.py
```

---

### 模型檔案

#### `multi_insurance_model.pkl` (2.8 MB)
**內容**:
- DQN 神經網路權重
- 19 個 XGBoost 分類器
- 資料標準化器 (Scaler)
- 特徵欄位名稱
- 保險產品清單

**生成方式**: 執行 `multi_insurance_model.py`

---

### 文檔

#### `README.md` (3.9 KB)
- 專案概述
- 快速開始指南
- 基本使用說明

#### `RECOMMENDATION_GUIDE.md` (9.9 KB)
- 詳細使用指南
- API 端點說明
- 輸入輸出格式
- 範例程式碼

#### `CROSS_SELL_SUMMARY.md` (7.4 KB)
- 系統功能總結
- 測試結果
- 支援的保險產品
- 模型效能

---

### 配置檔案

#### `requirements.txt` (336 B)
Python 依賴套件:
- flask (Web 框架)
- pandas, numpy (資料處理)
- scikit-learn (機器學習)
- xgboost (梯度提升)
- torch (深度學習)

**安裝**:
```bash
pip install -r requirements.txt
```

---

## 🗂️ 資料集

### `insurance+company+benchmark+coil+2000/`
CoIL Challenge 2000 資料集

#### 主要檔案:
- **ticdata2000.txt** (5,822 筆訓練資料)
  - 86 個欄位
  - 43 個客戶特徵欄位
  - 43 個保險相關欄位

- **dictionary.txt** - 欄位說明文檔

---

## 🔄 工作流程

### 1. 首次使用
```bash
# 安裝依賴
pip install -r requirements.txt

# 訓練模型
python multi_insurance_model.py

# 測試
python test_recommendation.py

# 啟動 API
python recommendation_app.py
```

### 2. 日常使用
```bash
# 直接啟動 API（模型已訓練）
python recommendation_app.py
```

### 3. 重新訓練
```bash
# 當有新資料時
python multi_insurance_model.py
```

---

## 📊 資料流向

```
訓練階段:
ticdata2000.txt
  → multi_insurance_model.py
  → multi_insurance_model.pkl

預測階段:
客戶資料
  → recommendation_app.py
  → multi_insurance_model.pkl
  → 推薦結果
```

---

## 💾 檔案大小

| 檔案 | 大小 | 說明 |
|------|------|------|
| multi_insurance_model.pkl | 2.8 MB | 訓練好的模型 |
| ticdata2000.txt | ~500 KB | 訓練資料 |
| multi_insurance_model.py | 14 KB | 訓練腳本 |
| recommendation_app.py | 11 KB | API 服務 |
| RECOMMENDATION_GUIDE.md | 9.9 KB | 使用指南 |
| CROSS_SELL_SUMMARY.md | 7.4 KB | 系統總結 |
| test_recommendation.py | 5.3 KB | 測試腳本 |
| README.md | 3.9 KB | 專案說明 |

**總計**: 約 3.3 MB

---

## 🔍 關鍵概念

### 模型架構
- **DQN** (Deep Q-Network): 提取 32 個高階特徵
- **XGBoost**: 19 個獨立分類器，每個預測一種保險
- **混合模式**: 結合深度學習和梯度提升的優勢

### 保險產品
系統支援 19 種保險產品預測:
- 車輛類: 汽車、機車、卡車等
- 財產類: 火災、財產、第三方責任險
- 人身類: 人壽、意外、殘疾保險
- 其他類: 自行車、船舶、社會保險

### 輸入特徵 (43 個)
- 人口統計: 年齡、家庭狀況
- 經濟狀況: 收入、購買力
- 社會特徵: 教育、職業、宗教
- 資產: 房屋、車輛所有權

---

## 📌 注意事項

1. **模型檔案**: `multi_insurance_model.pkl` 必須存在才能使用 API
2. **資料格式**: 客戶資料必須包含完整的 43 個特徵欄位
3. **Python 版本**: 建議使用 Python 3.8+
4. **記憶體**: 最少需要 2GB RAM

---

## 🔄 更新歷史

- **2025-10-16**: 初始版本
  - 建立多保險交叉銷售推薦系統
  - 訓練 19 種保險產品模型
  - 提供 RESTful API

---

## 📞 支援

詳細說明請參考:
- [README.md](README.md) - 快速開始
- [RECOMMENDATION_GUIDE.md](RECOMMENDATION_GUIDE.md) - 完整指南
- [CROSS_SELL_SUMMARY.md](CROSS_SELL_SUMMARY.md) - 功能總結
