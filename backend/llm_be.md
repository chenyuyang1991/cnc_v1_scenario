# CNC AI 優化器後端 API 文檔

## 概述

本文檔描述了 CNC AI 優化器的 FastAPI 後端實現，提供了完整的 API 端點來支援前端 Vue.js 應用程式的所有功能。後端採用模組化架構，包含認證、專案管理、場景處理、模擬執行、優化分析、檔案處理、聊天介面和配置管理等核心功能。

## 架構設計

### 目錄結構
```
backend/
├── main.py                 # FastAPI 應用程式入口點
├── requirements.txt        # Python 依賴套件
├── models/                 # Pydantic 資料模型
│   ├── auth.py            # 認證相關模型
│   ├── projects.py        # 專案管理模型
│   ├── scenarios.py       # 場景處理模型
│   ├── simulations.py     # 模擬執行模型
│   ├── optimization.py    # 優化分析模型
│   ├── files.py           # 檔案處理模型
│   ├── chat.py            # 聊天介面模型
│   └── config.py          # 配置管理模型
├── routers/               # API 路由處理器
│   ├── auth.py            # 認證端點
│   ├── projects.py        # 專案管理端點
│   ├── scenarios.py       # 場景處理端點
│   ├── simulations.py     # 模擬執行端點
│   ├── optimization.py    # 優化分析端點
│   ├── files.py           # 檔案處理端點
│   ├── chat.py            # 聊天介面端點
│   └── config.py          # 配置管理端點
├── services/              # 業務邏輯服務層
│   ├── auth_service.py    # 認證服務
│   ├── projects_service.py # 專案管理服務
│   ├── scenarios_service.py # 場景處理服務
│   ├── simulations_service.py # 模擬執行服務
│   ├── optimization_service.py # 優化分析服務
│   ├── files_service.py   # 檔案處理服務
│   ├── chat_service.py    # 聊天介面服務
│   └── config_service.py  # 配置管理服務
├── static/                # 靜態檔案
│   └── templates/         # 範本檔案
└── uploads/               # 上傳檔案暫存
```

### 技術棧
- **FastAPI**: 現代化的 Python Web 框架
- **Pydantic**: 資料驗證和序列化
- **python-jose**: JWT 令牌處理
- **passlib**: 密碼雜湊和驗證
- **aiofiles**: 非同步檔案操作
- **python-multipart**: 檔案上傳支援

## API 端點詳細說明

### 1. 認證系統 (`/auth`)

#### 登入端點
- **端點**: `POST /auth/login`
- **功能**: 使用者認證和 JWT 令牌生成
- **業務邏輯**: 
  - 驗證使用者憑證（使用者名稱和密碼）
  - 使用 bcrypt 進行密碼雜湊驗證
  - 生成 JWT 存取令牌（30分鐘有效期）
  - 返回使用者資訊和令牌
- **模擬資料**: 提供 `admin` 和 `operator` 兩個測試帳戶
- **前端連接**: 對應 `LoginPage.vue` 組件的登入表單

#### 登出端點
- **端點**: `POST /auth/logout`
- **功能**: 使用者登出處理
- **業務邏輯**: 返回登出成功訊息（實際應用中可加入令牌黑名單）

#### 令牌驗證端點
- **端點**: `GET /auth/verify`
- **功能**: 驗證 JWT 令牌有效性
- **業務邏輯**: 檢查令牌格式和有效期

### 2. 專案管理 (`/projects`)

#### 專案列表端點
- **端點**: `GET /projects`
- **功能**: 獲取所有專案列表
- **業務邏輯**: 返回包含專案 ID、名稱、描述、建立時間和狀態的專案清單
- **模擬資料**: 航太零件、汽車零件、醫療器材等三個範例專案
- **前端連接**: 對應 `ProjectSelectionModal.vue` 的專案下拉選單

#### 專案 CRUD 操作
- **端點**: `GET/POST/PUT/DELETE /projects/{id}`
- **功能**: 完整的專案生命週期管理
- **業務邏輯**: 
  - 建立新專案時自動生成唯一 ID
  - 更新專案資訊
  - 軟刪除專案（標記為非活躍狀態）

### 3. 場景處理 (`/scenarios`)

#### 場景管理
- **端點**: `GET/POST/PUT /scenarios`
- **功能**: CNC 加工場景的建立和管理
- **業務邏輯**:
  - 場景包含專案關聯、版本控制、完成度追蹤
  - 支援場景狀態管理（建立中、執行中、已完成）
- **模擬資料**: 汽車零件加工、航空零件製造、精密模具等場景
- **前端連接**: 對應 `IterationSelection.vue` 的場景選擇介面

#### 場景迭代端點
- **端點**: `POST /scenarios/{id}/iterate`
- **功能**: 基於現有場景建立迭代版本
- **業務邏輯**:
  - 支援參數、材料、刀具、策略四種迭代類型
  - 自動版本號遞增
  - 建立父子場景關聯
- **前端連接**: 對應迭代工作流程和 `IterationTypeModal.vue`

### 4. 模擬執行 (`/simulations`)

#### 模擬管理
- **端點**: `GET/POST /simulations`
- **功能**: CNC 加工模擬的建立和管理
- **業務邏輯**:
  - 模擬配置包含批次大小、品質檢查、測試類型等參數
  - 支援多種模擬狀態（待處理、執行中、已完成、失敗）
- **模擬資料**: 批次處理、品質測試、效能測試、負載測試等場景
- **前端連接**: 對應 `SimulationTable.vue` 的模擬列表和管理介面

#### 模擬執行端點
- **端點**: `POST /simulations/{id}/run`
- **功能**: 啟動模擬執行
- **業務邏輯**: 更新模擬狀態為執行中，返回執行確認

#### 模擬結果端點
- **端點**: `GET /simulations/{id}/results`
- **功能**: 獲取模擬執行結果
- **業務邏輯**: 提供執行時間、成功率、效能指標等詳細結果

### 5. 優化分析 (`/optimization`)

#### 優化執行端點
- **端點**: `POST /optimization/run`
- **功能**: 執行 CNC 程式優化分析
- **業務邏輯**:
  - 接收專案 ID、配置參數和檔案清單
  - 模擬 2 秒處理時間（實際應用中為複雜的優化演算法）
  - 生成唯一的優化 ID 用於結果追蹤
- **前端連接**: 對應 `ConfigurationSection.vue` 的優化執行按鈕

#### 優化結果端點
- **端點**: `GET /optimization/{id}/results`
- **功能**: 獲取詳細的優化分析結果
- **業務邏輯**: 提供五個主要結果類別：
  - **摘要數據**: 時間減少 23%、刀具壽命延長 15%、品質分數 98.5%、成本節省 $127
  - **圖表數據**: 加工時間比較、刀具磨損分析的數值資料
  - **程式碼差異**: 原始與優化後的 G-Code 對比，標示具體變更
  - **模擬資料**: 3D 模型 URL、動畫資料、總時間和步驟數
  - **驗證結果**: 安全檢查和品質檢查的通過狀態
- **前端連接**: 對應 `ResultsSection.vue` 的五個標籤頁內容

#### 優化配置端點
- **端點**: `GET /optimization/config`
- **功能**: 獲取優化配置選項
- **業務邏輯**: 提供機台、材料、刀具、優化目標和安全限制的預設配置

### 6. 檔案處理 (`/files`)

#### 檔案上傳端點
- **端點**: `POST /files/upload`
- **功能**: 處理 CNC 程式檔案上傳
- **業務邏輯**:
  - 支援多種檔案格式（.nc, .cnc, .gcode, .xlsx, .csv）
  - 非同步檔案儲存到 uploads 目錄
  - 生成唯一檔案 ID 和元資料
  - 記錄檔案大小、類型和上傳時間
- **前端連接**: 對應 `FileUploadSection.vue` 的檔案上傳功能

#### 檔案驗證端點
- **端點**: `POST /files/{fileId}/validate`
- **功能**: 驗證上傳檔案的有效性
- **業務邏輯**:
  - 檢查檔案格式支援性
  - 檔案大小警告（超過 10MB）
  - 估算處理時間和程式行數
  - 返回錯誤和警告訊息

#### 範本下載端點
- **端點**: `GET /files/templates/{type}`
- **功能**: 提供 Excel 和 CSV 範本檔案下載
- **業務邏輯**: 動態生成範本檔案，包含刀具、直徑、轉速、進給、深度等欄位

### 7. 聊天介面 (`/chat`)

#### 訊息處理端點
- **端點**: `POST /chat/send`
- **功能**: 處理使用者聊天訊息並生成 AI 回應
- **業務邏輯**:
  - 基於關鍵字分析生成上下文相關回應
  - 智能觸發 UI 組件顯示（檔案上傳、配置、結果）
  - 支援中文關鍵字識別和回應
  - 模擬 1 秒處理延遲以提供真實體驗
- **AI 回應邏輯**:
  - **開始/新專案**: 觸發檔案上傳介面
  - **配置/設定**: 觸發配置介面
  - **優化/執行**: 觸發結果顯示介面
  - **迭代/修改**: 提供迭代建議
- **前端連接**: 對應 `ChatInterface.vue` 的聊天功能

#### 聊天歷史端點
- **端點**: `GET /chat/history/{sessionId}`
- **功能**: 獲取指定會話的聊天歷史
- **業務邏輯**: 維護會話狀態和訊息歷史記錄

### 8. 配置管理 (`/config`)

#### 機台配置端點
- **端點**: `GET /config/machine`
- **功能**: 獲取可用機台配置
- **業務邏輯**: 提供 Haas VF-2、Mazak VTC-200、DMG Mori NHX-4000 等機台規格
- **配置內容**: 最大主軸轉速、最大進給速度、工作包絡尺寸

#### 材料配置端點
- **端點**: `GET /config/materials`
- **功能**: 獲取材料屬性配置
- **業務邏輯**: 提供鋁合金、碳鋼、不鏽鋼的物理屬性和建議參數
- **配置內容**: 硬度、密度、熱傳導率、建議轉速範圍

#### 刀具配置端點
- **端點**: `GET /config/tooling`
- **功能**: 獲取刀具配置選項
- **業務邏輯**: 提供端銑刀（平底/球頭）、面銑刀的規格和材料相容性
- **配置內容**: 可用直徑、材料相容性、建議參數倍數

#### 配置儲存端點
- **端點**: `POST /config/{type}`
- **功能**: 儲存使用者配置
- **業務邏輯**: 接收並儲存機台、材料、刀具等配置資料

## 模擬資料策略

### 真實性原則
所有模擬資料都基於實際 CNC 加工行業的標準和最佳實踐：

1. **機台規格**: 使用真實製造商的機台型號和規格
2. **材料屬性**: 基於工程材料手冊的實際物理屬性
3. **加工參數**: 符合行業標準的轉速、進給速度範圍
4. **優化結果**: 反映實際優化可能達到的改善幅度
5. **G-Code 範例**: 使用標準 G-Code 指令和格式

### 資料一致性
確保前後端資料模型完全一致：

1. **ID 格式**: 統一使用 `PRJ-001`、`SCN-001`、`SIM-001` 等格式
2. **時間格式**: 統一使用 ISO 8601 格式
3. **狀態值**: 前後端使用相同的狀態枚舉值
4. **數值精度**: 保持一致的小數位數和單位

## 錯誤處理機制

### HTTP 狀態碼
- **200**: 成功請求
- **201**: 成功建立資源
- **400**: 請求參數錯誤
- **401**: 認證失敗
- **404**: 資源不存在
- **500**: 伺服器內部錯誤

### 錯誤回應格式
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "錯誤描述",
    "details": "詳細信息"
  }
}
```

### 常見錯誤代碼
- `AUTH_REQUIRED`: 需要認證
- `INVALID_CREDENTIALS`: 無效憑證
- `FILE_TOO_LARGE`: 檔案過大
- `INVALID_FILE_FORMAT`: 無效檔案格式
- `OPTIMIZATION_FAILED`: 優化失敗
- `SIMULATION_ERROR`: 模擬錯誤

## 與前端的整合

### API 服務層對應
後端端點與前端 `apiService.js` 完全對應：

1. **authAPI**: 對應 `/auth` 路由
2. **projectAPI**: 對應 `/projects` 路由
3. **scenarioAPI**: 對應 `/scenarios` 路由
4. **simulationAPI**: 對應 `/simulations` 路由
5. **optimizationAPI**: 對應 `/optimization` 路由
6. **fileAPI**: 對應 `/files` 路由
7. **chatAPI**: 對應 `/chat` 路由
8. **configAPI**: 對應 `/config` 路由

### 組件資料流
每個 Vue 組件都有對應的後端支援：

1. **LandingPage.vue**: 靜態頁面，無需後端
2. **LoginPage.vue**: 使用 `/auth/login` 端點
3. **Sidebar.vue**: 使用認證狀態和導航邏輯
4. **ChatInterface.vue**: 使用 `/chat/send` 端點
5. **FileUploadSection.vue**: 使用 `/files/upload` 和 `/files/templates` 端點
6. **ConfigurationSection.vue**: 使用 `/config/*` 端點
7. **ResultsSection.vue**: 使用 `/optimization/results` 端點
8. **SimulationTable.vue**: 使用 `/simulations` 端點
9. **IterationSelection.vue**: 使用 `/scenarios` 端點
10. **ProjectSelectionModal.vue**: 使用 `/projects` 和 `/config/machine` 端點
11. **IterationTypeModal.vue**: 使用 `/scenarios/iterate` 端點

## 部署和開發

### 本地開發環境
1. 安裝依賴: `pip install -r requirements.txt`
2. 啟動服務: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
3. API 文檔: `http://localhost:8000/docs`
4. 健康檢查: `http://localhost:8000/health`

### CORS 配置
支援前端開發伺服器的跨域請求：
- `http://localhost:3000` (React 預設)
- `http://localhost:5173` (Vite 預設)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

### 靜態檔案服務
- 範本檔案: `/static/templates/`
- 上傳檔案: `uploads/` 目錄

## 擴展性考慮

### 資料庫整合
當前使用記憶體儲存，可輕鬆擴展至：
- PostgreSQL (推薦用於生產環境)
- SQLite (適合小型部署)
- MongoDB (適合文檔型資料)

### 認證增強
- OAuth 2.0 整合
- 多因子認證
- 角色基礎存取控制 (RBAC)

### 效能優化
- Redis 快取層
- 非同步任務佇列 (Celery)
- 檔案儲存服務 (AWS S3, MinIO)

### 監控和日誌
- 結構化日誌記錄
- 效能監控 (Prometheus)
- 錯誤追蹤 (Sentry)

## 安全性考慮

### 認證安全
- JWT 令牌有效期限制
- 密碼雜湊使用 bcrypt
- 安全的密鑰管理

### 檔案上傳安全
- 檔案類型驗證
- 檔案大小限制
- 病毒掃描整合

### API 安全
- 請求速率限制
- 輸入驗證和清理
- SQL 注入防護

## 結論

本後端實現提供了完整的 CNC AI 優化器功能支援，與前端 Vue.js 應用程式無縫整合。採用模組化架構設計，易於維護和擴展。所有 API 端點都經過精心設計，提供真實的模擬資料和完整的錯誤處理機制。

透過本文檔，開發團隊可以清楚了解每個端點的業務邏輯、資料流向和前後端整合方式，為後續的功能擴展和維護提供了堅實的基礎。
