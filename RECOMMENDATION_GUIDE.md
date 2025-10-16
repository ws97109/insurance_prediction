# 保險交叉銷售推薦系統使用指南

## 📋 系統概述

這是一個**智能保險交叉銷售推薦系統**，可以根據客戶的基本資料和已購買的保險，預測客戶可能購買的其他保險產品。

### 核心功能
- ✅ **多標籤預測**：同時預測 22 種保險產品的購買機率
- ✅ **個人化推薦**：根據客戶特徵提供量身定制的保險建議
- ✅ **交叉銷售**：識別客戶未購買但高機率會購買的保險
- ✅ **批量處理**：支援一次處理多個客戶的推薦

---

## 🎯 支援的 22 種保險產品

| 代碼 | 保險名稱 | 代碼 | 保險名稱 |
|------|---------|------|---------|
| WAPART | 私人第三方責任險 | LEVEN | 人壽保險 |
| WABEDR | 企業第三方責任險 | PERSONG | 個人意外險 |
| WALAND | 農業第三方責任險 | GEZONG | 家庭意外險 |
| PERSAUT | 汽車保險 | WAOREG | 殘疾保險 |
| BESAUT | 送貨車保險 | BRAND | 火災保險 |
| MOTSCO | 機車/速克達保險 | ZEILPL | 衝浪板保險 |
| VRAAUT | 卡車保險 | PLEZIER | 船舶保險 |
| AANHANG | 拖車保險 | FIETS | 自行車保險 |
| TRACTOR | 拖拉機保險 | INBOED | 財產保險 |
| WERKT | 農機保險 | BYSTAND | 社會保險 |
| BROM | 輕型機車保險 | CARAVAN | 旅遊保險 |

---

## 🚀 快速開始

### 步驟 1: 訓練模型

```bash
python multi_insurance_model.py
```

這會訓練所有 22 種保險的預測模型，並生成 `multi_insurance_model.pkl` 檔案。

**預計時間**: 5-10 分鐘

### 步驟 2: 啟動推薦 API

```bash
python recommendation_app.py
```

API 會在 `http://localhost:5000` 啟動。

---

## 📊 輸入資料格式

### 必要欄位：客戶基本資料（43 個欄位）

```json
{
  "MOSTYPE": 8,      // 客戶類型 (1-41)
  "MAANTHUI": 1,     // 房屋數量
  "MGEMOMV": 3,      // 平均家庭規模
  "MGEMLEEF": 3,     // 平均年齡 (1=20-30, 2=30-40, ...)
  "MOSHOOFD": 3,     // 客戶主要類型
  "MGODRK": 2,       // 天主教徒比例
  "MGODPR": 3,       // 新教徒比例
  "MGODOV": 1,       // 其他宗教比例
  "MGODGE": 2,       // 無宗教比例
  "MRELGE": 5,       // 已婚比例
  "MRELSA": 2,       // 同居比例
  "MRELOV": 1,       // 其他關係比例
  "MFALLEEN": 3,     // 單身比例
  "MFGEKIND": 4,     // 無子女家庭比例
  "MFWEKIND": 3,     // 有子女家庭比例
  "MOPLHOOG": 2,     // 高教育程度比例
  "MOPLMIDD": 4,     // 中教育程度比例
  "MOPLLAAG": 3,     // 低教育程度比例
  "MBERHOOG": 2,     // 高階層比例
  "MBERZELF": 1,     // 企業家比例
  "MBERBOER": 0,     // 農民比例
  "MBERMIDD": 3,     // 中階管理比例
  "MBERARBG": 4,     // 技術工人比例
  "MBERARBO": 2,     // 非技術工人比例
  "MSKA": 3,         // 社會階層 A
  "MSKB1": 4,        // 社會階層 B1
  "MSKB2": 3,        // 社會階層 B2
  "MSKC": 2,         // 社會階層 C
  "MSKD": 1,         // 社會階層 D
  "MHHUUR": 2,       // 租屋比例
  "MHKOOP": 5,       // 自有房屋比例
  "MAUT1": 5,        // 1輛車比例
  "MAUT2": 2,        // 2輛車比例
  "MAUT0": 1,        // 無車比例
  "MZFONDS": 4,      // 國民健保比例
  "MZPART": 3,       // 私人健保比例
  "MINKM30": 2,      // 收入<30K比例
  "MINK3045": 4,     // 收入30-45K比例
  "MINK4575": 3,     // 收入45-75K比例
  "MINK7512": 1,     // 收入75-122K比例
  "MINK123M": 0,     // 收入>123K比例
  "MINKGEM": 4,      // 平均收入
  "MKOOPKLA": 5      // 購買力等級
}
```

### 選填：已購買的保險

```json
{
  "AWAPART": 1,    // 已擁有 1 份私人第三方責任險
  "APERSAUT": 2,   // 已擁有 2 份汽車保險
  "ABRAND": 1      // 已擁有 1 份火災保險
}
```

---

## 🔧 API 使用方式

### 1. 單一客戶推薦

**端點**: `POST /api/recommend`

**請求範例**:
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "張三",
    "customer_features": {
      "MOSTYPE": 8,
      "MAANTHUI": 1,
      "MGEMOMV": 3,
      ... (其他 40 個欄位)
    },
    "owned_insurance": {
      "AWAPART": 1,
      "APERSAUT": 1
    }
  }'
```

**回應範例**:
```json
{
  "status": "success",
  "customer_name": "張三",
  "owned_insurance_count": 2,
  "owned_insurance": [
    {
      "product_code": "WAPART",
      "product_name": "私人第三方責任險"
    },
    {
      "product_code": "PERSAUT",
      "product_name": "汽車保險"
    }
  ],
  "recommendations": [
    {
      "product_code": "LEVEN",
      "product_name": "人壽保險",
      "probability": 0.85,
      "confidence": "high"
    },
    {
      "product_code": "BRAND",
      "product_name": "火災保險",
      "probability": 0.72,
      "confidence": "high"
    },
    {
      "product_code": "INBOED",
      "product_name": "財產保險",
      "probability": 0.68,
      "confidence": "medium"
    }
  ],
  "timestamp": "2025-10-16T14:30:00"
}
```

### 2. 批量客戶推薦

**端點**: `POST /api/recommend_batch`

#### 方式 A: 上傳 CSV 檔案

```bash
curl -X POST http://localhost:5000/api/recommend_batch \
  -F "file=@customers.csv"
```

CSV 檔案格式:
```csv
customer_name,MOSTYPE,MAANTHUI,MGEMOMV,...,AWAPART,APERSAUT
張三,8,1,3,...,1,1
李四,12,2,4,...,0,2
王五,6,1,2,...,1,0
```

#### 方式 B: JSON 陣列

```bash
curl -X POST http://localhost:5000/api/recommend_batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "customer_name": "張三",
      "MOSTYPE": 8,
      ...
    },
    {
      "customer_name": "李四",
      "MOSTYPE": 12,
      ...
    }
  ]'
```

**回應範例**:
```json
{
  "status": "success",
  "total_customers": 3,
  "successful_predictions": 3,
  "results": [
    {
      "customer_name": "張三",
      "owned_insurance_count": 2,
      "owned_insurance": [...],
      "top_recommendations": [
        {
          "product_code": "LEVEN",
          "product_name": "人壽保險",
          "probability": 0.85,
          "confidence": "high"
        },
        ...
      ]
    },
    ...
  ]
}
```

### 3. 取得保險產品列表

**端點**: `GET /api/products`

```bash
curl http://localhost:5000/api/products
```

### 4. 取得範例資料

**端點**: `GET /api/example`

```bash
curl http://localhost:5000/api/example
```

---

## 📈 輸出說明

### 推薦結果包含：

1. **customer_name**: 客戶姓名
2. **owned_insurance**: 已購買的保險列表
3. **recommendations**: 推薦購買的保險（前 5 項）
   - `product_code`: 保險代碼
   - `product_name`: 保險名稱（中文）
   - `probability`: 購買機率 (0-1)
   - `confidence`: 信心等級
     - `high`: probability ≥ 0.7
     - `medium`: 0.5 ≤ probability < 0.7
     - `low`: probability < 0.5

### 推薦邏輯

系統會：
1. ✅ 排除客戶已購買的保險
2. ✅ 只推薦購買機率 > 0.3 的產品
3. ✅ 按購買機率從高到低排序
4. ✅ 預設返回前 5 項推薦

---

## 🧪 測試範例

### Python 測試腳本

```python
import requests
import json

# 準備客戶資料
customer_data = {
    "customer_name": "測試客戶",
    "customer_features": {
        "MOSTYPE": 8,
        "MAANTHUI": 1,
        "MGEMOMV": 3,
        "MGEMLEEF": 3,
        "MOSHOOFD": 3,
        # ... 其他 38 個欄位設為 0 或適當值
    },
    "owned_insurance": {
        "AWAPART": 1,
        "APERSAUT": 1
    }
}

# 發送請求
response = requests.post(
    'http://localhost:5000/api/recommend',
    json=customer_data
)

# 顯示結果
result = response.json()
print(f"客戶: {result['customer_name']}")
print(f"已購保險: {result['owned_insurance_count']} 項")
print("\n推薦購買:")
for rec in result['recommendations']:
    print(f"  - {rec['product_name']}: {rec['probability']:.2%} ({rec['confidence']})")
```

---

## 📁 檔案結構

```
insurance_perdiction/
├── multi_insurance_model.py       # 多標籤預測模型
├── recommendation_app.py          # 推薦 API 服務
├── multi_insurance_model.pkl      # 訓練好的模型（執行訓練後產生）
├── RECOMMENDATION_GUIDE.md        # 本文件
└── insurance+company+benchmark+coil+2000/
    └── ticdata2000.txt           # 訓練資料
```

---

## ❓ 常見問題

### Q1: 模型需要多久訓練一次？
**A**: 當有新的客戶資料時，建議重新訓練模型以提升準確度。

### Q2: 如何提高推薦準確度？
**A**:
- 提供完整的客戶基本資料
- 準確填寫已購買的保險資訊
- 使用更多訓練資料重新訓練模型

### Q3: 可以調整推薦數量嗎？
**A**: 可以，修改 `recommendation_app.py` 中的 `result['recommendations'][:5]` 改為其他數字。

### Q4: 如何解讀購買機率？
**A**:
- **0.7+**: 強烈推薦，客戶很可能購買
- **0.5-0.7**: 中等推薦，有一定購買可能
- **0.3-0.5**: 低推薦，購買可能性較低
- **<0.3**: 不推薦

---

## 🔄 系統架構

```
客戶資料 (43 個特徵)
    ↓
DQN 特徵提取 (提取 32 個高階特徵)
    ↓
組合特徵 (43 + 32 = 75 個特徵)
    ↓
22 個 XGBoost 模型 (每個保險產品一個模型)
    ↓
預測每個產品的購買機率
    ↓
過濾已購買 & 低機率產品
    ↓
排序並返回 Top 5 推薦
```

---

## 📞 技術支援

如有問題，請檢查：
1. 模型是否已訓練 (`multi_insurance_model.pkl` 是否存在)
2. API 是否正常運行 (`GET /api/health`)
3. 輸入資料格式是否正確 (`GET /api/example` 查看範例)

---

## 🎉 開始使用

1. 訓練模型: `python multi_insurance_model.py`
2. 啟動 API: `python recommendation_app.py`
3. 測試推薦: 使用上述 API 端點

**祝您使用愉快！**
