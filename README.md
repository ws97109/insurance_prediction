# 保險交叉銷售推薦系統

智能保險推薦系統，根據客戶資料和已購保險，預測並推薦最適合的保險產品。

## 📋 功能特色

- ✅ **智能推薦**：基於 AI 模型預測 22 種保險產品
- ✅ **個人化**：根據客戶特徵量身定制推薦
- ✅ **交叉銷售**：自動識別未購買但高機率購買的保險
- ✅ **批量處理**：支援單一客戶和批量推薦
- ✅ **RESTful API**：易於整合到現有系統

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 訓練模型

```bash
python multi_insurance_model.py
```

訓練完成後會生成 `multi_insurance_model.pkl` (約 2.8 MB)

### 3. 測試系統

```bash
python test_recommendation.py
```

### 4. 啟動 API 服務

```bash
python recommendation_app.py
```

API 會在 `http://localhost:5000` 啟動

## 📊 系統說明

### 輸入

- **客戶名稱** (選填)
- **客戶基本資料** (43 個欄位)：年齡、收入、家庭狀況、教育程度、職業等
- **已購買保險** (選填)：系統會自動識別

### 輸出

```json
{
  "customer_name": "張三",
  "owned_insurance": [
    {"product_code": "PERSAUT", "product_name": "汽車保險"}
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

## 🔧 API 使用

### 單一客戶推薦

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "張三",
    "customer_features": {...},
    "owned_insurance": {...}
  }'
```

### 批量推薦 (CSV)

```bash
curl -X POST http://localhost:5000/api/recommend_batch \
  -F "file=@customers.csv"
```

### 取得範例資料

```bash
curl http://localhost:5000/api/example
```

## 📦 支援的保險產品 (19 種)

| 保險名稱 | 代碼 |
|---------|------|
| 私人第三方責任險 | WAPART |
| 汽車保險 | PERSAUT |
| 火災保險 | BRAND |
| 人壽保險 | LEVEN |
| 自行車保險 | FIETS |
| 機車保險 | MOTSCO |
| ... | ... |

完整列表請參考 [CROSS_SELL_SUMMARY.md](CROSS_SELL_SUMMARY.md)

## 📁 檔案結構

```
insurance_perdiction/
├── multi_insurance_model.py      # 模型訓練腳本
├── recommendation_app.py         # API 服務
├── test_recommendation.py        # 測試腳本
├── multi_insurance_model.pkl     # 訓練好的模型
├── requirements.txt              # Python 依賴
├── RECOMMENDATION_GUIDE.md       # 完整使用指南
├── CROSS_SELL_SUMMARY.md         # 系統總結
└── insurance+company+benchmark+coil+2000/
    └── ticdata2000.txt          # 訓練資料
```

## 🎯 使用場景

1. **客服中心**：即時推薦保險產品
2. **行銷活動**：批量生成客戶推薦名單
3. **網站個人化**：顯示個人化保險推薦
4. **業務工具**：輔助業務員推銷

## 📖 詳細文檔

- [RECOMMENDATION_GUIDE.md](RECOMMENDATION_GUIDE.md) - 完整使用指南
- [CROSS_SELL_SUMMARY.md](CROSS_SELL_SUMMARY.md) - 系統功能總結

## 🔬 技術架構

```
客戶資料 → DQN 特徵提取 → XGBoost 預測 → 推薦排序
```

- **DQN**: 深度 Q 網絡提取高階特徵
- **XGBoost**: 19 個分類器預測各保險購買機率
- **混合架構**: 結合深度學習和梯度提升

## 📊 模型效能

- 訓練樣本：5,822 客戶
- 保險產品：19 種
- 最佳 AUC：0.91 (殘疾保險)
- 平均 AUC：0.5-0.7

## 🛠️ 系統需求

- Python 3.8+
- 2GB RAM (最低)
- 100MB 磁碟空間

## 📝 授權

此專案僅供學習和研究使用。

## 🤝 貢獻

歡迎提交問題和改進建議。

---

**建立日期**: 2025-10-16
**版本**: 1.0
**作者**: Insurance Prediction Team

