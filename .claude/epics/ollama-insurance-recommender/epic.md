---
name: ollama-insurance-recommender
status: backlog
created: 2025-10-16T10:16:53Z
progress: 0%
prd: .claude/prds/ollama-insurance-recommender.md
github: [將在同步到 GitHub 時更新]
---

# Epic: Ollama Insurance Recommender

## Overview

建立整合 Ollama Qwen2.5:7b 的智能保險推薦系統，為現有的多保險預測系統增加深度客戶分析和銷售話術生成能力。系統將接收 ML 模型的預測結果，透過 LLM 分析客戶特徵，自動生成購買原因、痛點分析和個人化銷售腳本，幫助業務人員提升成交率。

**技術路徑**：擴展現有 Flask API，整合本地 Ollama 服務，新增前端頁面展示 AI 分析結果。

---

## Architecture Decisions

### 1. 整合策略：擴展而非重寫
**決策**：擴展現有的 `recommendation_app.py`，而非建立新服務
**理由**：
- 重用現有的 ML 模型載入和預測邏輯
- 避免重複的 API 結構和配置
- 降低維護成本和系統複雜度

### 2. Ollama 客戶端：使用 requests 庫
**決策**：直接使用 Python `requests` 呼叫 Ollama HTTP API
**理由**：
- Ollama 提供標準的 REST API
- 避免額外的 SDK 依賴
- 簡單、穩定、易於調試

### 3. 提示詞管理：模板化設計
**決策**：將提示詞存為 JSON/YAML 模板檔案
**理由**：
- 業務人員可以修改提示詞而無需改程式碼
- 支援 A/B 測試不同提示詞版本
- 易於國際化和客製化

### 4. 快取策略：檔案系統快取
**決策**：使用簡單的檔案快取（pickle/JSON）
**理由**：
- 避免引入 Redis 等額外依賴
- 本地部署環境簡單
- 對於 1000 用戶規模足夠

### 5. 前端技術：伺服器端渲染 (SSR)
**決策**：使用 Jinja2 模板 + Bootstrap，而非 React/Vue
**理由**：
- 降低前端複雜度
- 無需 build 流程
- 符合現有技術棧
- 更快的首屏載入

### 6. 相似案例：簡化為餘弦相似度
**決策**：使用 scikit-learn 的餘弦相似度，而非複雜的向量資料庫
**理由**：
- 資料量小（< 10,000 案例）
- 計算成本可接受
- 避免引入 Elasticsearch/Milvus 等重型依賴

---

## Technical Approach

### Frontend Components

#### 1. 擴展現有推薦介面
- **修改**：`recommendation_app.py` 的回應格式，新增 `ollama_analysis` 欄位
- **新增**：前端 JavaScript 動態展開 Ollama 分析面板
- **展示內容**：
  - 購買原因分析（卡片式）
  - 客戶痛點列表（標籤雲）
  - 銷售話術（可複製的文字框）
  - 相似案例縮圖（可點擊展開）

#### 2. 新增批量分析頁面
- **路由**：`/batch-analysis`
- **功能**：CSV 上傳 → 批量呼叫 Ollama → 下載報告
- **進度條**：使用 WebSocket 或 polling 顯示進度

#### 3. 系統狀態頁面（可選）
- **路由**：`/system-status`
- **展示**：Ollama 連接狀態、快取命中率、平均回應時間

### Backend Services

#### 1. Ollama 服務層
**新檔案**：`ollama/ollama_service.py`

```python
class OllamaService:
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model = model_name

    def analyze_customer(self, customer_data, prediction):
        """生成客戶分析"""
        prompt = self._build_prompt(customer_data, prediction)
        response = self._call_ollama(prompt)
        return self._parse_response(response)

    def generate_sales_script(self, customer_data, analysis):
        """生成銷售話術"""
        # ...
```

#### 2. 提示詞管理器
**新檔案**：`ollama/prompt_manager.py`

```python
class PromptManager:
    def load_template(self, template_name):
        """從 YAML 載入提示詞模板"""
        # ...

    def render_prompt(self, template, **kwargs):
        """填充變數生成最終提示詞"""
        # ...
```

#### 3. 快取管理器
**新檔案**：`utils/cache_manager.py`

```python
class CacheManager:
    def get_cached_analysis(self, cache_key):
        """查詢快取"""
        # ...

    def save_analysis(self, cache_key, result, ttl=3600):
        """儲存快取"""
        # ...
```

#### 4. 相似案例搜尋
**新檔案**：`utils/case_matcher.py`

```python
class CaseMatcher:
    def find_similar_cases(self, customer_features, top_k=3):
        """使用餘弦相似度找相似案例"""
        # 從 SQLite 或檔案載入歷史案例
        # 計算相似度
        # 返回 top K 案例
```

#### 5. API 端點擴展
**修改**：`recommendation_app.py`

新增路由：
- `POST /api/recommend` - 擴展回應，新增 `ollama_analysis`
- `POST /api/ollama/batch` - 批量分析
- `GET /api/ollama/status` - Ollama 狀態檢查

### Infrastructure

#### 1. 目錄結構重組
```
insurance_prediction/
├── models/                     # 現有 ML 模型
│   └── multi_insurance_model.pkl
├── ollama/                     # 新增 Ollama 相關
│   ├── __init__.py
│   ├── ollama_service.py       # Ollama 服務層
│   ├── prompt_manager.py       # 提示詞管理
│   ├── prompts/                # 提示詞模板
│   │   ├── analysis.yaml
│   │   └── sales_script.yaml
│   └── config.yaml             # Ollama 配置
├── utils/                      # 工具函數
│   ├── cache_manager.py        # 快取管理
│   └── case_matcher.py         # 案例匹配
├── web/                        # 前端資源
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       ├── index.html          # 現有首頁（修改）
│       ├── batch.html          # 新增批量頁面
│       └── components/         # 共用組件
│           └── ollama_analysis.html
├── recommendation_app.py       # 主應用（修改）
├── multi_insurance_model.py    # 現有模型（不變）
├── config.yaml                 # 全域配置（新增）
└── requirements.txt            # 依賴（新增 pyyaml）
```

#### 2. 配置管理
**新檔案**：`config.yaml`

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:7b"
  timeout: 60
  temperature: 0.7

cache:
  enabled: true
  directory: ".cache/ollama"
  ttl: 3600

api:
  port: 5000
  debug: false
```

#### 3. 部署考量
- **依賴檢查**：啟動時檢查 Ollama 服務是否可用
- **降級方案**：Ollama 不可用時，仍返回基本推薦（不包含分析）
- **日誌**：記錄 Ollama 呼叫時間、錯誤等

---

## Implementation Strategy

### 開發原則
1. **漸進式**：先基本功能，再進階特性
2. **可測試**：每個模組獨立可測試
3. **可配置**：硬編碼最小化，使用配置檔案
4. **錯誤處理**：優雅降級，不因 Ollama 故障導致整體崩潰

### 風險緩解
1. **Ollama 效能**：
   - 實現積極快取（快取命中率目標 > 40%）
   - 提供「快速模式」選項（簡化提示詞）
   - 提供 timeout 設定

2. **內容品質**：
   - 提供多版本提示詞 A/B 測試
   - 記錄用戶回饋（點讚/點踩）
   - 定期人工審核生成內容

3. **整合問題**：
   - 充分的錯誤處理和重試機制
   - Fallback 到基本推薦模式
   - 詳細的日誌記錄

### 測試策略
1. **單元測試**：每個服務類獨立測試
2. **整合測試**：Mock Ollama API 測試端到端流程
3. **手動測試**：業務團隊實際使用並提供回饋
4. **效能測試**：模擬 100 並發請求

---

## Task Breakdown Preview

系統將分解為以下 **8 個核心任務**（符合 ≤10 任務的要求）：

### Phase 1: 基礎建設（Week 1）
- [ ] **Task 1**: 專案結構重組與配置管理
  - 建立新目錄結構（ollama/, web/）
  - 建立 config.yaml 和配置載入邏輯
  - 更新 requirements.txt

- [ ] **Task 2**: Ollama 服務整合
  - 實現 OllamaService 類（連接、呼叫、錯誤處理）
  - 實現 PromptManager（模板載入、渲染）
  - 建立基礎提示詞模板（analysis.yaml, sales_script.yaml）
  - 實現快取管理器（CacheManager）

### Phase 2: 核心功能（Week 2）
- [ ] **Task 3**: 擴展 API 支援 Ollama 分析
  - 修改 `/api/recommend` 端點整合 Ollama
  - 實現分析邏輯（購買原因、痛點、話術）
  - 實現降級邏輯（Ollama 不可用時）

- [ ] **Task 4**: 相似案例推薦
  - 實現 CaseMatcher 類（餘弦相似度）
  - 建立案例儲存結構（SQLite 或 JSON 檔案）
  - 整合到推薦流程

### Phase 3: 前端介面（Week 3）
- [ ] **Task 5**: 前端頁面開發
  - 修改現有推薦頁面，新增 Ollama 分析展示區
  - 實現動態展開/收合面板
  - 實現話術複製功能
  - 美化 UI（Bootstrap 卡片、標籤雲）

- [ ] **Task 6**: 批量分析功能
  - 新增 `/batch-analysis` 頁面（CSV 上傳）
  - 實現批量處理 API（/api/ollama/batch）
  - 實現進度顯示（polling 或 WebSocket）
  - 實現報告下載

### Phase 4: 優化與測試（Week 4）
- [ ] **Task 7**: 效能優化與測試
  - 實現快取機制並驗證命中率
  - 進行效能測試（回應時間、並發）
  - 優化提示詞（根據測試結果）
  - 撰寫單元測試

- [ ] **Task 8**: 文檔與部署
  - 撰寫使用文檔（README、使用指南）
  - 建立 Ollama 安裝指南
  - 建立故障排除文檔
  - 準備 UAT 環境和培訓材料

---

## Dependencies

### 外部依賴
1. **Ollama** (必須)
   - 版本：0.1.x+
   - 用途：運行 Qwen2.5:7b 模型
   - 安裝：`curl -fsSL https://ollama.com/install.sh | sh`
   - 驗證：`ollama --version`

2. **Qwen2.5:7b 模型** (必須)
   - 下載：`ollama pull qwen2.5:7b`
   - 大小：約 4.7GB

3. **Python 套件** (新增)
   ```
   pyyaml>=6.0       # 配置檔案解析
   requests>=2.31.0  # Ollama API 呼叫
   ```

### 內部依賴
1. **現有系統**
   - `multi_insurance_model.pkl` - 必須存在且可載入
   - `MultiInsurancePredictor.predict_for_customer()` - 必須正常運作

2. **硬體環境**
   - CPU: 4 核心以上（建議 8 核心）
   - RAM: 16GB 以上（建議 32GB）
   - 磁碟：> 10GB 可用空間
   - GPU: 可選（NVIDIA CUDA 支援可大幅提升速度）

---

## Success Criteria (Technical)

### 功能完成度
- ✅ Ollama 整合：可正常呼叫並接收回應
- ✅ 分析品質：生成內容符合業務需求（人工評估 > 85% 滿意度）
- ✅ 話術生成：包含開場白、說服點、結案技巧三部分
- ✅ 相似案例：相關性 > 80%（人工評估）
- ✅ 網頁界面：操作流暢，無明顯 bug

### 效能指標
- ✅ 單一分析回應時間：< 30 秒（95 分位）
- ✅ 快取命中率：> 40%
- ✅ 系統可用性：> 99%（測試期間）
- ✅ 並發支援：100 用戶同時使用無錯誤

### 程式碼品質
- ✅ 單元測試覆蓋率：> 70%
- ✅ 程式碼格式化：通過 Black 檢查
- ✅ 文檔完整：README、API 文檔、部署指南

### 使用者體驗
- ✅ 頁面載入：< 2 秒
- ✅ 錯誤訊息：清晰友善
- ✅ 操作指引：新用戶可獨立完成操作
- ✅ 繁體中文：所有介面和生成內容使用繁體中文

---

## Estimated Effort

### 總體時程
- **Phase 1 (基礎建設)**：1 週
- **Phase 2 (核心功能)**：1 週
- **Phase 3 (前端介面)**：1 週
- **Phase 4 (優化測試)**：1 週
- **總計**：4 週（1 名全職開發者）

### 任務工作量估算
| 任務 | 工作量 | 優先級 | 風險 |
|------|--------|--------|------|
| Task 1: 專案結構重組 | 0.5 天 | P0 | 低 |
| Task 2: Ollama 服務整合 | 2 天 | P0 | 中 |
| Task 3: API 擴展 | 2 天 | P0 | 中 |
| Task 4: 相似案例推薦 | 1.5 天 | P1 | 低 |
| Task 5: 前端頁面開發 | 3 天 | P0 | 低 |
| Task 6: 批量分析功能 | 2 天 | P1 | 中 |
| Task 7: 效能優化與測試 | 2 天 | P0 | 中 |
| Task 8: 文檔與部署 | 1 天 | P0 | 低 |

**總工作量**：14 天（2 週實際開發 + 2 週緩衝和測試）

### 關鍵路徑
```
Task 1 → Task 2 → Task 3 → Task 5 → Task 7 → Task 8
```

**關鍵路徑任務必須按順序完成，其他任務可並行。**

### 資源需求
- **開發人員**：1 名（Python + Web 開發）
- **測試人員**：業務團隊 3-5 人（UAT）
- **硬體**：1 台測試伺服器（建議規格如依賴章節）
- **外部支援**：IT 團隊協助 Ollama 安裝和部署

---

## 簡化說明

相較於 PRD 的完整規劃，本 Epic 做了以下簡化：

1. **減少檔案結構層級**：不建立過深的 api/routes/controllers/services 結構，直接在主檔案擴展
2. **簡化快取**：使用檔案系統而非 Redis
3. **簡化相似案例**：使用餘弦相似度而非向量資料庫
4. **伺服器端渲染**：使用 Jinja2 而非前後端分離架構
5. **統計儀表板**：標記為可選，第一版不實現
6. **合併相關任務**：將密切相關的功能合併為單一任務

**目標**：在保證核心功能完整的前提下，降低系統複雜度，加快交付速度。

---

## Next Steps

完成 Epic 建立後，執行以下指令分解任務：

```bash
/pm:epic-decompose ollama-insurance-recommender
```

這將為每個任務建立獨立的實作檔案和驗收標準。
