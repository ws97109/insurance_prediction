---
name: insurance
status: backlog
created: 2025-10-09T02:47:18Z
progress: 0%
prd: .claude/prds/insurance.md
github: [Will be updated when synced to GitHub]
---

# Epic: Insurance AI Prediction System

## Overview

建構一個基於深度強化學習與梯度提升的保險智能推薦系統。系統核心採用 **DQN (Deep Q-Network)** 進行交叉銷售預測，搭配 **XGBoost** 執行客戶價值分群。透過 FastAPI 後端與 React 前端提供視覺化儀表板，為保險業務團隊提供數據驅動的客戶洞察與產品推薦。

**技術棧精簡策略**：
- 資料層：使用 Jupyter Notebook + pandas 進行 EDA 與特徵工程
- 模型層：TensorFlow/PyTorch (DQN) + XGBoost (分群)
- API 層：FastAPI (輕量、自動生成 OpenAPI 文件)
- 前端層：React + Chart.js (避免過度複雜的 D3.js)
- 部署：Docker Compose (開發/測試) → 雲端容器服務 (生產)
- 資料庫：PostgreSQL (結構化資料) + Redis (快取)

## Architecture Decisions

### AD-1: DQN vs 監督式學習 (Mixed Approach)
**決策**：初期同時開發 DQN 與 XGBoost 監督式學習作為交叉銷售預測的備案
**理由**：
- DQN 理論上可透過強化學習優化長期收益，但訓練複雜度高
- 若 DQN 準確率未達 75% 或訓練困難，可快速切換至 XGBoost 多分類模型
- 降低技術風險，確保專案如期交付

**落地方案**：
- Week 3-6 並行開發 DQN 與 XGBoost (交叉銷售版本)
- Week 7 進行模型對比評估，選擇最佳方案

### AD-2: 簡化架構 - 避免過度工程
**決策**：MVP 階段採用單體架構，延後微服務化
**理由**：
- 團隊規模小 (3 人)，微服務會增加複雜度
- 單體架構開發速度快，易於調試
- 效能需求 (50 並發) 單體架構即可滿足

**架構設計**：
```
Docker Compose:
  - app (FastAPI: API + 模型推論服務)
  - db (PostgreSQL)
  - redis (快取)
  - nginx (反向代理)
```

### AD-3: 前端技術選型 - React + Chart.js
**決策**：使用 React 搭配 Chart.js，而非 D3.js
**理由**：
- Chart.js 開箱即用，支援響應式圖表，開發速度快
- D3.js 學習曲線陡峭，對前端工程師要求高
- 業務需求為基本圖表 (圓餅圖、柱狀圖、折線圖)，Chart.js 已足夠

### AD-4: 模型訓練環境分離
**決策**：訓練與推論環境分離
**理由**：
- 訓練需要 GPU，推論可使用 CPU
- 降低生產環境資源成本
- 訓練可在本地或 Colab/Kaggle 執行，推論服務部署至雲端

**實作方式**：
- 訓練腳本：本地 Jupyter Notebook 或雲端訓練平台
- 模型產出：儲存為 .h5 (TensorFlow) / .pkl (XGBoost)
- 推論服務：載入預訓練模型，僅執行 inference

### AD-5: API 設計 - RESTful 優先
**決策**：使用 RESTful API，暫不導入 GraphQL
**理由**：
- RESTful 簡單易懂，團隊熟悉度高
- FastAPI 自動生成 OpenAPI 文件，便於前端整合
- GraphQL 過度複雜，不符合專案需求

**核心端點**：
- `POST /api/v1/predict/single` - 單筆客戶預測
- `POST /api/v1/predict/batch` - 批量預測 (非同步)
- `POST /api/v1/segment` - 客戶分群
- `GET /api/v1/models/metrics` - 模型效能指標

### AD-6: 資料處理策略
**決策**：使用 pandas + scikit-learn 進行資料前處理，避免使用 Spark
**理由**：
- COIL 2000 資料集僅 5822 + 4000 筆，pandas 足以處理
- Spark 部署與維護成本高，不符合成本效益
- pandas 生態成熟，團隊熟悉度高

### AD-7: 安全性實作 - 分階段強化
**決策**：
- **Phase 1 (MVP)**：基本 JWT 認證 + HTTPS
- **Phase 3 (上線前)**：加密儲存 + RBAC + 稽核日誌

**理由**：避免過早優化，先確保核心功能完成

## Technical Approach

### Frontend Components

#### 核心頁面 (3 個主要頁面)
1. **Dashboard - 客戶分群儀表板**
   - 元件：PieChart (價值分群占比)、BarChart (群組統計)、FilterPanel (篩選器)
   - 狀態管理：React Context API (避免引入 Redux)
   - 資料流：useEffect → API call → setState → 圖表渲染

2. **Recommendation - 客戶推薦頁面**
   - 元件：SearchBox (客戶 ID 輸入)、RecommendationCard (推薦產品卡片)、ExportButton
   - 互動：即時搜尋 (debounce 300ms) → API 請求 → 顯示 Top 3 推薦

3. **ModelManagement - 模型管理介面**
   - 元件：MetricsChart (準確率趨勢)、UploadForm (訓練資料上傳)、TrainingProgress
   - 狀態：WebSocket 或輪詢顯示訓練進度

#### UI 框架選擇
- **Material-UI (MUI)** 或 **Ant Design**：快速建構企業級 UI
- 響應式設計：使用 CSS Grid + Flexbox，確保手機/平板相容

### Backend Services

#### API 服務層 (FastAPI)
```python
# 主要模組結構
app/
├── api/
│   ├── v1/
│   │   ├── predict.py      # 預測端點
│   │   ├── segment.py      # 分群端點
│   │   ├── models.py       # 模型管理端點
├── core/
│   ├── config.py           # 配置管理
│   ├── security.py         # JWT 認證
├── models/
│   ├── dqn_inference.py    # DQN 推論邏輯
│   ├── xgboost_inference.py # XGBoost 推論邏輯
│   ├── loader.py           # 模型載入器
├── schemas/
│   ├── prediction.py       # Pydantic 模型
├── db/
│   ├── models.py           # SQLAlchemy ORM
│   ├── crud.py             # 資料庫操作
├── main.py                 # FastAPI 應用入口
```

#### 關鍵實作細節

**1. 模型載入與快取**
```python
# 單例模式載入模型，避免重複載入
class ModelLoader:
    _dqn_model = None
    _xgb_model = None

    @classmethod
    def get_dqn_model(cls):
        if cls._dqn_model is None:
            cls._dqn_model = load_model('models/dqn_model.h5')
        return cls._dqn_model
```

**2. 批量預測非同步處理**
```python
# 使用 Celery 或 FastAPI BackgroundTasks
@app.post("/api/v1/predict/batch")
async def batch_predict(file: UploadFile, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_batch, task_id, file)
    return {"task_id": task_id, "status": "processing"}
```

**3. Redis 快取策略**
- 快取客戶預測結果 (TTL: 24 小時)
- 快取鍵格式：`predict:{customer_id}:{model_version}`

#### 資料模型設計 (PostgreSQL)

**customers** 表
```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    features JSONB NOT NULL,  -- 儲存 86 維特徵向量
    created_at TIMESTAMP DEFAULT NOW()
);
```

**predictions** 表
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    predicted_product VARCHAR(50),
    probability FLOAT,
    model_version VARCHAR(20),
    predicted_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

**segments** 表
```sql
CREATE TABLE segments (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    segment VARCHAR(10),  -- high/medium/low
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

### Infrastructure

#### 開發環境 (Docker Compose)
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/insurance
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=insurance
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000

volumes:
  postgres_data:
```

#### 生產環境部署策略
- **選項 1 (推薦)**：AWS ECS Fargate + RDS + ElastiCache
  - 優點：Serverless 容器，自動擴展，無需管理伺服器
  - 成本：中等 (Pay-as-you-go)

- **選項 2**：GCP Cloud Run + Cloud SQL + Memorystore
  - 優點：快速部署，自動 HTTPS，擴展至零
  - 成本：低 (按請求計費)

- **選項 3**：Azure Container Instances + Azure Database for PostgreSQL
  - 優點：整合 Azure 生態

**推薦架構 (AWS)**：
```
Internet → ALB (HTTPS) → ECS Fargate (FastAPI) → RDS PostgreSQL
                                ↓
                          ElastiCache Redis
```

#### 監控與日誌
- **應用監控**：Prometheus (metrics) + Grafana (dashboard)
- **日誌**：CloudWatch Logs / Stackdriver Logging
- **告警**：CloudWatch Alarms / GCP Alerting
- **模型追蹤**：MLflow (實驗追蹤、模型版本管理)

## Implementation Strategy

### 簡化開發策略 (6-7 個月)

**關鍵簡化點**：
1. **減少過度設計**：不實作 A/B 測試框架、複雜的 CI/CD pipeline (初期使用基本 GitHub Actions)
2. **重用開源工具**：不自建訓練排程系統，使用 Jupyter Notebook + 手動訓練
3. **延後次要功能**：PDF 匯出、Webhook、多語言 (僅繁中)
4. **精簡測試**：核心邏輯單元測試 80%+，端到端測試 (E2E) 僅覆蓋關鍵流程

### 開發階段規劃

#### Phase 1: 資料 + 模型 (Week 1-8)
**Week 1-2: 資料工程**
- 任務：EDA → 特徵工程 → 訓練/測試分割
- 風險：資料品質問題 (缺失值、不平衡)
- 產出：清洗後的資料集 + Notebook 報告

**Week 3-6: 模型開發**
- 並行開發：DQN + XGBoost (交叉銷售備案)
- 實驗管理：使用 MLflow 追蹤實驗
- 產出：訓練好的模型檔案 (.h5, .pkl)

**Week 7-8: XGBoost 客戶分群**
- 任務：定義分群標籤 → 訓練 → 超參數調整
- 產出：分群模型 + 特徵重要性報告

#### Phase 2: API + 前端 (Week 9-16)
**Week 9-11: FastAPI 開發**
- 任務：API 端點實作 → 模型整合 → 單元測試
- 產出：可運行的 API 服務 + Swagger 文件

**Week 12-14: React 前端**
- 任務：UI 元件開發 → API 串接 → 響應式設計
- 產出：可操作的前端應用

**Week 15-16: 整合測試**
- 任務：端到端測試 → 效能測試 → Bug 修復
- 產出：整合後的完整系統

#### Phase 3: 優化 + 上線 (Week 17-24)
**Week 17-18: 模型優化**
- 任務：超參數精調 → 對比實驗
- 產出：優化後的模型版本

**Week 19-20: 安全性 + 合規**
- 任務：HTTPS/TLS → 資料加密 → GDPR 審查
- 產出：安全性檢查清單 + 合規報告

**Week 21-22: UAT**
- 任務：業務人員測試 → 收集回饋 → 修正
- 產出：UAT 報告 + Bug 修正清單

**Week 23-24: 部署上線**
- 任務：雲端部署 → 監控配置 → 使用者培訓
- 產出：生產環境運行系統

### 風險緩解策略

**風險 1: DQN 訓練困難或準確率不達標**
- 緩解：並行開發 XGBoost 監督式學習作為備案
- 判斷時間點：Week 6 結束前決定採用哪個模型

**風險 2: 資料品質問題**
- 緩解：Week 1-2 深入 EDA，提前發現問題
- 備案：若 COIL 2000 不適用，使用合成資料或其他公開資料集

**風險 3: 前端開發延遲**
- 緩解：API 優先開發，前端使用 Mock API 並行
- 簡化：使用 UI 框架 (MUI/Ant Design) 加速開發

**風險 4: 部署複雜度**
- 緩解：使用 Docker Compose 統一開發/測試/生產環境
- 簡化：選擇 Managed Service (如 Cloud Run) 降低運維負擔

### 測試策略

**單元測試 (80% 覆蓋率目標)**
- 後端：pytest (API 端點、模型推論邏輯)
- 前端：Jest + React Testing Library (關鍵元件)

**整合測試**
- API 整合測試：使用 httpx 測試端到端 API 流程
- 資料庫測試：使用測試資料庫 (Docker)

**效能測試**
- 工具：Locust / Apache JMeter
- 目標：50 並發使用者，API 回應時間 < 3 秒

**UAT 測試**
- 對象：15-20 位業務人員
- 時間：2 週
- 方法：提供測試案例清單，收集回饋

## Task Breakdown Preview

為了符合「10 個任務以內」的要求，我們將任務高度整合：

- [ ] **Task 1: 資料準備與特徵工程** (Week 1-2)
  - 載入 COIL 2000 資料集
  - EDA 分析與資料清洗
  - 特徵工程與資料分割
  - 產出：清洗後的訓練/測試資料集

- [ ] **Task 2: 模型開發與訓練** (Week 3-8)
  - DQN 模型開發 (含 Experience Replay、Target Network)
  - XGBoost 交叉銷售模型 (備案)
  - XGBoost 客戶分群模型
  - 模型評估與選擇
  - 產出：訓練好的模型檔案 (.h5, .pkl)

- [ ] **Task 3: FastAPI 後端開發** (Week 9-11)
  - API 端點實作 (predict/single, predict/batch, segment)
  - 模型載入與推論邏輯
  - 資料庫 ORM 與 CRUD 操作
  - Redis 快取整合
  - 單元測試 (80% 覆蓋率)
  - 產出：FastAPI 應用 + Swagger 文件

- [ ] **Task 4: React 前端開發** (Week 12-14)
  - 客戶分群儀表板 (圓餅圖、柱狀圖)
  - 客戶推薦頁面 (搜尋、顯示 Top 3)
  - 模型管理介面 (效能監控)
  - 響應式設計 (RWD)
  - 產出：可操作的前端應用

- [ ] **Task 5: 整合測試與優化** (Week 15-16)
  - 端到端整合測試
  - 效能測試與優化 (API 回應時間、頁面載入速度)
  - Docker Compose 配置
  - 產出：整合後的完整系統

- [ ] **Task 6: 模型優化與實驗** (Week 17-18)
  - DQN 超參數精調
  - XGBoost 特徵選擇優化
  - 模型對比實驗 (MLflow 追蹤)
  - 產出：優化後的模型版本

- [ ] **Task 7: 安全性與合規實作** (Week 19-20)
  - HTTPS/TLS 配置
  - JWT 認證機制
  - 資料加密 (AES-256)
  - GDPR 合規檢查
  - 產出：安全性檢查清單 + 合規報告

- [ ] **Task 8: UAT 使用者驗收測試** (Week 21-22)
  - 招募 15-20 位業務人員
  - 執行測試案例
  - 收集回饋與 Bug 修正
  - 產出：UAT 報告 + Bug 修正清單

- [ ] **Task 9: 雲端部署與監控配置** (Week 23)
  - 選擇雲端平台 (AWS/GCP/Azure)
  - 容器部署 (ECS Fargate / Cloud Run)
  - 資料庫遷移 (RDS / Cloud SQL)
  - Prometheus + Grafana 監控
  - 產出：生產環境運行系統

- [ ] **Task 10: 使用者培訓與上線** (Week 24)
  - 撰寫操作手冊
  - 錄製教學影片
  - 業務團隊培訓 (2 小時)
  - 正式上線
  - 產出：培訓教材 + 上線報告

## Dependencies

### 外部依賴
- **COIL 2000 資料集**：已公開於 UCI Repository，無可用性風險
- **雲端平台**：需提前 1 個月申請資源 (AWS/GCP/Azure)
- **GPU 資源**：可使用 Google Colab / Kaggle Kernel 作為備援

### 內部依賴
- **業務團隊**：需求確認 (Week 1)、UAT 測試 (Week 21-22)
- **IT 基礎設施團隊**：伺服器資源申請 (Week 17)
- **資訊安全團隊**：安全審查 (Week 19-20)
- **法務團隊**：GDPR 合規審查 (Week 20)

### 技術依賴
- Python 3.8+
- TensorFlow 2.x / PyTorch 1.x (DQN)
- XGBoost 1.5+
- FastAPI 0.95+
- React 18+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose

### 依賴風險
- **高風險**：資料品質問題、DQN 訓練困難、安全審查失敗
- **中風險**：業務團隊採用率低、API 效能不達標
- **低風險**：第三方套件相容性問題

## Success Criteria (Technical)

### 模型效能指標
- ✅ 交叉銷售預測準確率 ≥ 75% (測試集)
- ✅ 客戶分群準確率 ≥ 85% (測試集)
- ✅ Top 3 推薦命中率 ≥ 85%
- ✅ 模型訓練可重現性 100%

### 系統效能指標
- ✅ 單筆預測回應時間 < 3 秒 (95th percentile)
- ✅ 批量預測 (500 筆) < 10 分鐘
- ✅ 儀表板頁面載入時間 < 2 秒
- ✅ 系統可用性 ≥ 99%
- ✅ 同時支援 50+ 並發使用者

### 程式碼品質指標
- ✅ 單元測試覆蓋率 ≥ 80%
- ✅ API 文件完整性 100% (Swagger UI)
- ✅ 符合 PEP 8 規範 (Python)
- ✅ 無 Critical 等級安全漏洞

### 業務指標 (上線後 3 個月)
- ✅ 業務經理週活躍使用率 ≥ 90%
- ✅ 業務員週活躍使用率 ≥ 70%
- ✅ 平均每週預測查詢量 ≥ 1000 次
- ✅ 使用者滿意度 ≥ 4.0/5.0

### 交付物檢查清單
- ✅ 訓練好的模型檔案 (DQN + XGBoost)
- ✅ FastAPI 應用 + Swagger 文件
- ✅ React 前端應用 (響應式設計)
- ✅ Docker Compose 配置
- ✅ 部署指南文件
- ✅ 操作手冊 + 教學影片
- ✅ 單元測試報告
- ✅ UAT 測試報告
- ✅ 安全性檢查清單
- ✅ 生產環境運行系統

## Estimated Effort

### 人力配置
- **ML 工程師** (1 位)：Week 1-18 (資料、模型、優化)
- **後端工程師** (1 位)：Week 9-24 (API、部署、監控)
- **前端工程師** (1 位)：Week 12-24 (UI、整合、優化)

### 總時程
- **Phase 1 (資料+模型)**：8 週
- **Phase 2 (API+前端)**：8 週
- **Phase 3 (優化+上線)**：8 週
- **總計**：24 週 (約 6 個月)

### 關鍵路徑
```
資料準備 (2週) → 模型開發 (6週) → API開發 (3週) → 前端開發 (3週) →
整合測試 (2週) → 優化 (2週) → 安全審查 (2週) → UAT (2週) → 部署上線 (2週)
```

**最長路徑**：資料準備 → 模型開發 → API 開發 (11 週)
**並行機會**：前端開發可在 API 開發期間使用 Mock API 並行

### 資源需求
- **開發環境**：本地 MacBook / Linux 工作站
- **訓練環境**：Google Colab Pro (GPU) 或 Kaggle Kernel
- **雲端資源**：AWS/GCP 中等規模實例 (預估 $300-500/月)
- **測試人員**：15-20 位業務人員 (UAT 階段)

### 風險緩衝
- 每個 Phase 預留 10% 時間緩衝 (約 1 週)
- 高風險任務 (DQN 訓練、安全審查) 預留額外 2 週 buffer
- 總時程彈性：6-7 個月

---

## 簡化亮點總結

相較於原 PRD 的 7 個月規劃，本 Epic 透過以下簡化策略縮短至 6 個月：

1. **架構簡化**：單體架構替代微服務，減少複雜度
2. **技術選型務實**：Chart.js 替代 D3.js，快速交付
3. **功能精簡**：延後 PDF 匯出、Webhook、多語言等次要功能
4. **並行開發**：前端使用 Mock API 與後端並行
5. **重用工具**：MLflow 替代自建實驗平台
6. **測試精簡**：聚焦核心功能，E2E 測試僅覆蓋關鍵流程
7. **部署簡化**：選擇 Managed Service (Cloud Run/Fargate) 降低運維成本

**核心原則**：先做能用的 MVP，再迭代優化，避免過度工程。
