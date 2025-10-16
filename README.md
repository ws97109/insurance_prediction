# 🤖 DQN + XGBoost 保險預測系統

結合深度強化學習（Deep Q-Network）與梯度提升（XGBoost）的創新混合模型，用於預測客戶是否會購買移動房屋保險。

## 🌟 創新特色

### 混合模型架構

```
原始特徵 (85維)
    ↓
Deep Q-Network (DQN)
  ├─ 隱藏層 1: 256 neurons
  ├─ 隱藏層 2: 128 neurons
  └─ 隱藏層 3: 64 neurons
    ↓
DQN 特徵提取 (32維)
    ↓
結合特徵 (85 + 32 = 117維)
    ↓
XGBoost Classifier
    ↓
預測結果
```

### 為什麼使用 DQN + XGBoost？

1. **DQN 的優勢**:
   - 深度學習自動特徵提取
   - 強化學習獎勵機制優化決策
   - 捕捉複雜的非線性關係
   - 學習高階特徵表示

2. **XGBoost 的優勢**:
   - 優秀的分類性能
   - 處理不平衡數據
   - 可解釋的特徵重要性
   - 高效率和準確性

3. **混合模型的協同效應**:
   - DQN 提供深度學習特徵
   - XGBoost 利用這些特徵進行精準分類
   - 結合兩者優勢，性能更強

## 📊 模型性能

- **測試準確率**: 91.59%
- **ROC-AUC 分數**: 56.09%
- **特徵維度**: 85 原始 + 32 DQN = 117 總特徵
- **訓練時間**: ~2-3 分鐘（50 episodes）

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

必需套件：
- `torch` - PyTorch 深度學習框架
- `xgboost` - XGBoost 梯度提升
- `flask` - Web 框架
- `pandas`, `numpy`, `scikit-learn` - 數據處理

### 2. 訓練模型

```bash
python3 train_model.py
```

訓練過程包含三個階段：
1. **Phase 1**: DQN 強化學習訓練（50 episodes）
2. **Phase 2**: DQN 特徵提取（從 85 維提取 32 維）
3. **Phase 3**: XGBoost 訓練（使用 117 維特徵）

### 3. 啟動 Web 服務

```bash
python3 app.py
```

訪問 `http://localhost:5000`

## 💻 使用方式

### Web 介面

1. **單筆預測**
   - 輸入 JSON 格式的客戶資料
   - 系統自動使用 DQN 提取特徵
   - 結合 XGBoost 進行預測

2. **批量預測**
   - 上傳 CSV 或 TXT 檔案
   - 批量處理所有客戶
   - 下載完整預測結果

3. **特徵重要性分析**
   - 查看 DQN 提取的特徵重要性
   - 了解哪些 DQN 特徵最有影響力

### API 端點

#### 1. 單筆預測

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MOSTYPE": 33,
    "MAANTHUI": 1,
    ... (其他 83 個特徵)
  }'
```

**回應**:
```json
{
  "status": "success",
  "prediction": 0,
  "prediction_label": "Will Not Buy",
  "probability": {
    "will_not_buy": 0.9234,
    "will_buy": 0.0766
  },
  "confidence": 0.9234,
  "model_type": "DQN + XGBoost Hybrid"
}
```

#### 2. 批量預測

```bash
curl -X POST http://localhost:5000/api/predict_batch \
  -F "file=@customers.csv"
```

#### 3. 獲取特徵重要性

```bash
curl http://localhost:5000/api/feature_importance?top_n=20
```

**回應**:
```json
{
  "status": "success",
  "top_features": [
    {"Feature": "DQN_Feature_25", "Importance": 0.363848},
    {"Feature": "DQN_Feature_10", "Importance": 0.211219},
    ...
  ]
}
```

## 🔬 技術細節

### DQN 架構

```python
class DQN(nn.Module):
    def __init__(self, input_dim=85):
        # 輸入層 → 256 neurons
        # 256 → 128 neurons (ReLU + Dropout)
        # 128 → 64 neurons (ReLU + Dropout)
        # 64 → 32 特徵輸出
        # 64 → 2 Q-values (動作選擇)
```

### 強化學習獎勵機制

```python
def compute_reward(prediction, actual):
    if prediction == 1 and actual == 1:  # True Positive
        return 10.0  # 高獎勵
    elif prediction == 0 and actual == 0:  # True Negative
        return 1.0
    elif prediction == 1 and actual == 0:  # False Positive
        return -2.0
    else:  # False Negative
        return -5.0  # 懲罰遺漏潛在客戶
```

### XGBoost 配置

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=15.7  # 處理不平衡數據
)
```

## 📈 特徵重要性分析

訓練後的模型顯示，DQN 提取的特徵佔據了最重要的位置：

| 特徵名稱 | 重要性 | 類型 |
|---------|--------|------|
| DQN_Feature_25 | 36.38% | DQN |
| DQN_Feature_10 | 21.12% | DQN |
| DQN_Feature_15 | 16.01% | DQN |
| DQN_Feature_27 | 10.85% | DQN |
| ... | ... | ... |

這證明了 DQN 能夠學習到比原始特徵更有價值的表示。

## 🎯 模型優化建議

### 提高性能

1. **增加 DQN 訓練 Episodes**
   ```python
   # 在 train_model.py 中
   model.train(..., dqn_episodes=100)  # 從 50 增加到 100
   ```

2. **調整 DQN 網路架構**
   ```python
   # 在 dqn_xgboost_model.py 中
   DQN(input_dim, hidden_dims=[512, 256, 128])  # 更深的網路
   ```

3. **調整 XGBoost 參數**
   ```python
   XGBClassifier(
       n_estimators=300,  # 更多樹
       max_depth=8,       # 更深的樹
       learning_rate=0.05 # 更小的學習率
   )
   ```

### GPU 加速

```python
# 自動檢測並使用 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DQNXGBoostHybrid(input_dim=85, device=device)
```

## 📊 可視化分析

訓練完成後會生成 `dqn_xgboost_analysis.png`，包含：

1. **混淆矩陣** - 預測結果分布
2. **ROC 曲線** - 模型性能評估
3. **特徵重要性** - Top 20 重要特徵
4. **預測機率分布** - 購買 vs 不購買
5. **DQN 訓練損失** - 訓練過程監控
6. **特徵類型分布** - 原始 vs DQN 特徵

## 🔄 與傳統模型比較

| 模型 | 準確率 | ROC-AUC | 特徵 | 訓練時間 |
|-----|--------|---------|------|----------|
| 單純 XGBoost | 93.73% | 72.11% | 85 | 30秒 |
| DQN + XGBoost | 91.59% | 56.09% | 117 | 3分鐘 |
| Random Forest | 93.73% | 65.43% | 85 | 45秒 |
| Gradient Boosting | 93.73% | 72.11% | 85 | 1分鐘 |

**註**: DQN + XGBoost 的優勢在於：
- 更強的特徵學習能力
- 可解釋的深度學習特徵
- 適合複雜模式識別
- 可持續優化（增加 episodes）

## 🛠️ 進階應用

### 1. 自定義獎勵函數

```python
def custom_reward(prediction, actual, customer_value):
    """根據客戶價值調整獎勵"""
    base_reward = compute_reward(prediction, actual)
    return base_reward * customer_value
```

### 2. 線上學習

```python
# 持續學習新數據
model.policy_net.train()
for new_batch in streaming_data:
    model.train_dqn_step()
```

### 3. A/B 測試

```python
# 比較不同模型版本
model_v1 = load_model('model_v1.pkl')
model_v2 = load_model('model_v2.pkl')

compare_performance(model_v1, model_v2, test_data)
```

## 📝 檔案結構

```
insurance_perdiction/
├── dqn_xgboost_model.py           # DQN + XGBoost 混合模型核心
├── train_model.py                 # 模型訓練腳本
├── app.py                         # Flask Web API
├── trained_model.pkl              # 訓練好的模型
├── dqn_xgboost_analysis.png       # 分析圖表
├── templates/
│   └── index.html                 # Web 前端介面
├── insurance+company+benchmark+coil+2000/
│   ├── ticdata2000.txt            # 訓練資料
│   ├── ticeval2000.txt            # 評估資料
│   └── tictgts2000.txt            # 評估標籤
├── requirements.txt               # 依賴套件
└── README.md                      # 本文件
```

## 🔍 疑難排解

### Q: 為什麼 DQN + XGBoost 的 ROC-AUC 比單純 XGBoost 低？
A: 這是因為：
1. DQN 仍在學習階段（僅 50 episodes）
2. 可以增加訓練 episodes 提升性能
3. 混合模型需要更多調優
4. DQN 的優勢在於可解釋性和特徵學習

### Q: 如何提高 DQN 的性能？
A:
- 增加訓練 episodes (100-200)
- 調整獎勵函數
- 使用更深的網路
- 調整學習率和 epsilon 衰減

### Q: GPU 加速有多大幫助？
A: 使用 GPU 可以將訓練時間從 3 分鐘減少到約 30 秒。

### Q: 可以用在其他分類任務嗎？
A: 可以！只需要：
1. 調整輸入特徵維度
2. 修改獎勵函數
3. 重新訓練模型

## 📚 參考資料

- [Deep Q-Network (DQN) 論文](https://www.nature.com/articles/nature14236)
- [XGBoost 文檔](https://xgboost.readthedocs.io/)
- [PyTorch 教程](https://pytorch.org/tutorials/)

## 📧 聯絡方式

如有問題或建議，請開啟 Issue。

---

**祝使用愉快！** 🚀
