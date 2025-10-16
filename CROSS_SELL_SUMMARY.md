# 保險交叉銷售推薦系統 - 完整總結

## ✅ 系統已建立完成

您的保險交叉銷售推薦系統已經成功建立並測試完成！

---

## 🎯 系統功能

### 輸入資訊
1. **客戶名稱** (選填)
2. **客戶基本資料** (43 個欄位)
   - 年齡、收入、家庭狀況
   - 教育程度、職業
   - 房屋、車輛擁有情況
3. **已購買的保險** (選填)
   - 系統會自動識別客戶已擁有的保險

### 輸出結果
1. **客戶名稱**
2. **已購買保險列表**
3. **未購買但高機率會購買的保險推薦**
   - 保險代碼 (如 LEVEN)
   - 保險名稱 (如 人壽保險)
   - 購買機率 (如 85%)
   - 信心等級 (高/中/低)

---

## 📁 建立的檔案

### 核心程式
1. **multi_insurance_model.py** - 多保險預測模型
   - 訓練 19 種保險產品的預測模型
   - DQN + XGBoost 混合架構

2. **recommendation_app.py** - 推薦 API 服務
   - RESTful API (Port 5000)
   - 支援單一客戶和批量推薦

3. **test_recommendation.py** - 測試腳本
   - 驗證系統功能
   - 展示使用範例

### 模型檔案
4. **multi_insurance_model.pkl** (2.8 MB)
   - 訓練好的模型
   - 包含 19 個保險產品預測器

### 文件
5. **RECOMMENDATION_GUIDE.md** - 完整使用指南
6. **CROSS_SELL_SUMMARY.md** - 本文件

---

## 🚀 快速使用

### 方式 1: 使用測試腳本 (簡單測試)

```bash
python3 test_recommendation.py
```

這會展示兩個客戶的推薦範例。

### 方式 2: 啟動 API 服務 (完整功能)

```bash
# 啟動 API
python3 recommendation_app.py

# API 會在 http://localhost:5000 啟動
```

然後使用以下 API：

#### 取得單一客戶推薦
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "張三",
    "customer_features": {
      "MOSTYPE": 8,
      "MAANTHUI": 1,
      ...
    },
    "owned_insurance": {
      "APERSAUT": 1,
      "ABRAND": 1
    }
  }'
```

#### 取得範例資料
```bash
curl http://localhost:5000/api/example
```

#### 批量推薦 (上傳 CSV)
```bash
curl -X POST http://localhost:5000/api/recommend_batch \
  -F "file=@customers.csv"
```

---

## 📊 測試結果範例

### 測試客戶 1: 張三 (中產家庭)
**已購保險**:
- 汽車保險
- 火災保險

**推薦購買**:
1. 🚴 **自行車保險** - 60.3% (中信心度)
2. 🛡️ **私人第三方責任險** - 45.5%
3. 🛵 **機車保險** - 31.9%

### 測試客戶 2: 李四 (富裕年輕家庭)
**已購保險**:
- 私人第三方責任險

**推薦購買**:
1. 🔥 **火災保險** - 58.7% (中信心度)
2. 🚗 **汽車保險** - 43.5%
3. 🚴 **自行車保險** - 35.0%

---

## 📋 支援的保險產品 (19 種)

系統成功訓練了以下保險產品的預測模型：

| 保險名稱 | 代碼 | 訓練樣本 |
|---------|------|---------|
| 私人第三方責任險 | WAPART | 2,340 客戶 |
| 汽車保險 | PERSAUT | 2,977 客戶 |
| 火災保險 | BRAND | 3,156 客戶 |
| 企業第三方責任險 | WABEDR | 82 客戶 |
| 農業第三方責任險 | WALAND | 120 客戶 |
| 送貨車保險 | BESAUT | 48 客戶 |
| 機車/速克達保險 | MOTSCO | 222 客戶 |
| 拖車保險 | AANHANG | 65 客戶 |
| 拖拉機保險 | TRACTOR | 143 客戶 |
| 農機保險 | WERKT | 21 客戶 |
| 輕型機車保險 | BROM | 396 客戶 |
| 人壽保險 | LEVEN | 293 客戶 |
| 個人意外險 | PERSONG | 31 客戶 |
| 家庭意外險 | GEZONG | 38 客戶 |
| 殘疾保險 | WAOREG | 23 客戶 |
| 船舶保險 | PLEZIER | 33 客戶 |
| 自行車保險 | FIETS | 147 客戶 |
| 財產保險 | INBOED | 45 客戶 |
| 社會保險 | BYSTAND | 82 客戶 |

**註**: 有 3 種保險因樣本數不足而未訓練 (VRAAUT, ZEILPL, CARAVAN)

---

## 🔍 模型效能

### 最佳表現的保險產品
- **殘疾保險 (WAOREG)**: AUC 0.91 ⭐⭐⭐⭐⭐
- **拖拉機保險 (TRACTOR)**: AUC 0.84 ⭐⭐⭐⭐
- **農機保險 (WERKT)**: AUC 0.83 ⭐⭐⭐⭐
- **農業責任險 (WALAND)**: AUC 0.76 ⭐⭐⭐

### 一般表現的保險產品
- 大多數保險: AUC 0.5-0.7 ⭐⭐⭐

---

## 🎓 使用場景

### 1. 客服中心推薦
當客服與客戶通話時，輸入客戶資料，即時獲得推薦。

### 2. 行銷活動
上傳客戶名單 CSV，批量生成推薦，用於精準行銷。

### 3. 網站個人化
整合到網站，為登入用戶顯示個人化保險推薦。

### 4. 業務員工具
業務員拜訪前，查詢客戶推薦，準備推銷話術。

---

## 🔧 系統架構

```
客戶資料 (43 個特徵)
    ↓
DQN 特徵提取
    ↓
組合特徵 (43 + 32 = 75)
    ↓
19 個 XGBoost 模型
    ↓
預測 19 種保險的購買機率
    ↓
過濾已購買保險
    ↓
排序並返回 Top 5 推薦
```

---

## 📊 輸出格式詳解

### JSON 回應結構
```json
{
  "status": "success",
  "customer_name": "張三",
  "owned_insurance_count": 2,
  "owned_insurance": [
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
    }
  ]
}
```

### 信心等級說明
- **high** (高): probability ≥ 0.7 - 強烈推薦
- **medium** (中): 0.5 ≤ probability < 0.7 - 中等推薦
- **low** (低): 0.3 ≤ probability < 0.5 - 弱推薦

---

## 📞 API 端點總覽

| 方法 | 端點 | 說明 |
|-----|------|------|
| GET | / | API 資訊 |
| GET | /api/products | 列出所有保險產品 |
| POST | /api/recommend | 單一客戶推薦 |
| POST | /api/recommend_batch | 批量客戶推薦 |
| GET | /api/example | 取得範例資料 |
| GET | /api/health | 健康檢查 |

---

## 💡 進階使用

### 調整推薦數量
編輯 `recommendation_app.py`:
```python
# 改為返回前 10 項推薦
'top_recommendations': result['recommendations'][:10]
```

### 調整推薦門檻
編輯 `multi_insurance_model.py`:
```python
# 只推薦高機率產品
if pred['product_code'] not in owned_codes and pred['probability'] > 0.5:
```

### 重新訓練模型
如有新資料，重新訓練：
```bash
python3 multi_insurance_model.py
```

---

## ✅ 下一步建議

1. **整合到現有系統**
   - 將 API 整合到 CRM 系統
   - 建立前端介面

2. **優化模型**
   - 收集更多客戶資料
   - 定期重新訓練模型

3. **擴展功能**
   - 加入推薦理由說明
   - 計算預期收益
   - A/B 測試不同推薦策略

4. **監控效果**
   - 追蹤推薦接受率
   - 分析哪些保險推薦最成功

---

## 🎉 總結

您的保險交叉銷售推薦系統已完全建立並測試完成！

### 核心優勢
✅ **智能推薦** - AI 驅動的個人化推薦
✅ **高效率** - 批量處理數千客戶
✅ **易整合** - RESTful API 設計
✅ **可擴展** - 支援新增保險產品

### 立即開始使用
```bash
# 測試系統
python3 test_recommendation.py

# 或啟動 API 服務
python3 recommendation_app.py
```

**詳細文件**: 參考 [RECOMMENDATION_GUIDE.md](RECOMMENDATION_GUIDE.md)

---

**建立日期**: 2025-10-16
**系統版本**: 1.0
**模型檔案**: multi_insurance_model.pkl (2.8 MB)
**訓練樣本**: 5,822 客戶
