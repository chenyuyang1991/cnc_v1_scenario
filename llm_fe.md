# CNC AI 優化器 - 前端模組與 API 對應文檔

## 概述

本文檔詳細描述了 CNC AI 優化器前端應用程式的模組化架構，包括每個 UI 組件、對應的後端 API 端點、以及預期的輸入輸出格式。

---

## 1. 登陸頁面模組 (LandingPage.vue)

### 組件位置
- **文件**: `frontend/src/components/LandingPage.vue`
- **功能**: 平台選擇入口
- **API 依賴**: 無

### UI 元素
- 平台選擇卡片 (v0/v1/data)
- 平台圖標和描述
- 懸停效果和動畫

### 事件處理
```javascript
// 輸出事件
@navigate(view, platform)
// view: 'login'
// platform: 'v0' | 'v1' | 'data'
```

### 狀態管理
- 使用 `selectedPlatform` 存儲選擇的平台

---

## 2. 登入頁面模組 (LoginPage.vue)

### 組件位置
- **文件**: `frontend/src/components/LoginPage.vue`
- **功能**: 用戶認證
- **API 依賴**: `/auth/login`

### UI 元素
- 用戶名輸入框
- 密碼輸入框
- 登入按鈕
- 返回按鈕

### API 對應

#### 登入 API
```javascript
// 端點: POST /auth/login
// 輸入格式
{
  "username": "string",
  "password": "string",
  "platform": "v0" | "v1" | "data"
}

// 預期輸出格式
{
  "success": true,
  "token": "jwt_token_string",
  "user": {
    "id": "user_id",
    "username": "string",
    "role": "admin" | "user",
    "permissions": ["read", "write", "admin"]
  },
  "platform": "v0" | "v1" | "data"
}
```

### 事件處理
```javascript
// 輸入事件
@login(credentials)
// credentials: { username, password, platform }

// 輸出事件
@navigate(view)
// view: 'landing'
```

---

## 3. 側邊欄模組 (Sidebar.vue)

### 組件位置
- **文件**: `frontend/src/components/Sidebar.vue`
- **功能**: 導航和模式切換
- **API 依賴**: `/auth/verify`

### UI 元素
- 應用程式標題和平台標識
- 登出按鈕
- 模式選擇按鈕 (新專案/迭代/模擬)

### API 對應

#### Token 驗證 API
```javascript
// 端點: GET /auth/verify
// 輸入: Authorization header with JWT token
// 預期輸出
{
  "valid": true,
  "user": {
    "id": "user_id",
    "username": "string",
    "role": "string"
  }
}
```

### 事件處理
```javascript
// 輸出事件
@logout()
@setMode(mode)
// mode: 'new' | 'iterate' | 'simulation'
```

---

## 4. 聊天界面模組 (ChatInterface.vue)

### 組件位置
- **文件**: `frontend/src/components/ChatInterface.vue`
- **功能**: AI 助手對話界面
- **API 依賴**: `/chat/send`, `/chat/history`

### UI 元素
- 聊天消息列表
- 消息輸入框
- 發送按鈕
- 連接狀態指示器

### API 對應

#### 發送消息 API
```javascript
// 端點: POST /chat/send
// 輸入格式
{
  "message": "string",
  "context": {
    "mode": "new" | "iterate" | "simulation",
    "scenario_id": "string",
    "session_id": "string"
  }
}

// 預期輸出格式
{
  "id": "message_id",
  "content": "string",
  "type": "assistant",
  "timestamp": "2024-01-15T10:30:00Z",
  "showFileUpload": boolean,
  "showConfig": boolean,
  "showResults": boolean,
  "metadata": {
    "suggestions": ["string"],
    "actions": ["upload", "configure", "optimize"]
  }
}
```

#### 聊天歷史 API
```javascript
// 端點: GET /chat/history/{sessionId}
// 預期輸出格式
{
  "session_id": "string",
  "messages": [
    {
      "id": "message_id",
      "type": "user" | "assistant",
      "content": "string",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### 事件處理
```javascript
// 輸入事件
@send-message(message)
@file-upload(event)
@file-remove(file)
@download-template(type)
@validate-files()
@close-config()
@run-optimization()
@close-results()
@implement-optimization()

// 輸出事件
// 所有事件都向上傳遞給父組件
```

---

## 5. 檔案上傳模組 (FileUploadSection.vue)

### 組件位置
- **文件**: `frontend/src/components/FileUploadSection.vue`
- **功能**: 檔案上傳和驗證
- **API 依賴**: `/files/upload`, `/files/validate`, `/files/templates`

### UI 元素
- 拖放上傳區域
- 檔案列表顯示
- 範本下載按鈕
- 驗證按鈕

### API 對應

#### 檔案上傳 API
```javascript
// 端點: POST /files/upload
// 輸入格式: FormData
{
  "file": File,
  "type": "cnc" | "excel" | "csv",
  "scenario_id": "string"
}

// 預期輸出格式
{
  "file_id": "string",
  "filename": "string",
  "size": number,
  "type": "string",
  "status": "uploaded",
  "validation_required": boolean
}
```

#### 檔案驗證 API
```javascript
// 端點: POST /files/{fileId}/validate
// 預期輸出格式
{
  "valid": boolean,
  "errors": [
    {
      "line": number,
      "message": "string",
      "severity": "error" | "warning"
    }
  ],
  "warnings": ["string"],
  "file_type": "gcode" | "excel" | "csv",
  "metadata": {
    "line_count": number,
    "tool_count": number,
    "estimated_time": number
  }
}
```

#### 範本下載 API
```javascript
// 端點: GET /files/templates/{type}
// type: "excel" | "csv"
// 預期輸出: 檔案流 (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
```

### 事件處理
```javascript
// 輸入事件
@file-upload(event)
@file-remove(file)
@download-template(type)
@validate-files()

// 輸出事件
// 向上傳遞給父組件
```

---

## 6. 配置面板模組 (ConfigurationSection.vue)

### 組件位置
- **文件**: `frontend/src/components/ConfigurationSection.vue`
- **功能**: 參數配置設定
- **API 依賴**: `/config/machine`, `/config/materials`, `/config/tooling`, `/config/save`

### UI 元素
- 標籤頁導航 (機台/材料/刀具/優化/安全)
- 配置表單
- 保存/取消按鈕

### API 對應

#### 機台配置 API
```javascript
// 端點: GET /config/machine
// 預期輸出格式
{
  "machines": [
    {
      "id": "cnc-001",
      "name": "Haas VF-2",
      "type": "milling",
      "specs": {
        "max_spindle_speed": 8000,
        "max_feed_rate": 1000,
        "work_area": "762x406x508"
      }
    }
  ]
}
```

#### 材料配置 API
```javascript
// 端點: GET /config/materials
// 預期輸出格式
{
  "materials": [
    {
      "id": "aluminum-6061",
      "name": "鋁合金 6061-T6",
      "density": 2.7,
      "hardness": 95,
      "cutting_speed": 300,
      "feed_rate": 0.1
    }
  ]
}
```

#### 刀具配置 API
```javascript
// 端點: GET /config/tooling
// 預期輸出格式
{
  "tools": [
    {
      "id": "endmill-6mm",
      "name": "端銑刀 6mm",
      "type": "end_mill",
      "diameter": 6,
      "flutes": 4,
      "material": "carbide"
    }
  ]
}
```

#### 保存配置 API
```javascript
// 端點: POST /config/{type}
// type: "machine" | "material" | "tooling" | "optimization" | "safety"
// 輸入格式
{
  "scenario_id": "string",
  "config": {
    "spindle_speed": 3000,
    "feed_rate": 500,
    "material_id": "aluminum-6061",
    "tool_id": "endmill-6mm",
    "optimization_goals": ["time", "quality"],
    "safety_limits": {
      "max_spindle_speed": 5000,
      "max_feed_rate": 1000
    }
  }
}

// 預期輸出格式
{
  "config_id": "string",
  "status": "saved",
  "validation": {
    "valid": true,
    "warnings": ["string"]
  }
}
```

### 事件處理
```javascript
// 輸入事件
@close()
@run-optimization()

// 輸出事件
// 向上傳遞給父組件
```

---

## 7. 結果展示模組 (ResultsSection.vue)

### 組件位置
- **文件**: `frontend/src/components/ResultsSection.vue`
- **功能**: 優化結果展示
- **API 依賴**: `/optimization/results`, `/optimization/export`

### UI 元素
- 結果標籤頁 (摘要/圖表/程式碼/模擬/驗證)
- 數據視覺化區域
- 匯出按鈕

### API 對應

#### 優化結果 API
```javascript
// 端點: GET /optimization/{id}/results
// 預期輸出格式
{
  "optimization_id": "string",
  "status": "completed",
  "summary": {
    "time_reduction": -23,
    "tool_life_improvement": 15,
    "quality_score": 98.5,
    "cost_savings": 127
  },
  "charts": {
    "time_comparison": {
      "type": "bar",
      "data": [
        {"label": "原始", "value": 45.2},
        {"label": "優化", "value": 34.8}
      ]
    },
    "tool_wear": {
      "type": "line",
      "data": [/* time series data */]
    }
  },
  "code_diff": {
    "original": "G90 G54 G17...",
    "optimized": "G90 G54 G17...",
    "changes": [
      {
        "line": 4,
        "original": "S3000 M3",
        "optimized": "S3500 M3"
      }
    ]
  },
  "simulation": {
    "3d_model_url": "string",
    "animation_data": "string",
    "collision_check": true
  },
  "validation": {
    "safety": {
      "passed": true,
      "checks": ["collision", "speed_limits"]
    },
    "quality": {
      "passed": true,
      "surface_roughness": 0.8,
      "dimensional_accuracy": 0.02
    }
  }
}
```

#### 匯出結果 API
```javascript
// 端點: POST /optimization/{id}/export
// 輸入格式
{
  "format": "pdf" | "excel" | "json",
  "sections": ["summary", "charts", "code", "simulation"]
}

// 預期輸出: 檔案流
```

### 事件處理
```javascript
// 輸入事件
@close()
@implement-optimization()

// 輸出事件
// 向上傳遞給父組件
```

---

## 8. 模擬表格模組 (SimulationTable.vue)

### 組件位置
- **文件**: `frontend/src/components/SimulationTable.vue`
- **功能**: 模擬列表管理
- **API 依賴**: `/simulations`, `/simulations/export`, `/simulations/import`

### UI 元素
- 模擬列表表格
- 操作按鈕 (建立/匯出/匯入)
- 狀態指示器

### API 對應

#### 模擬列表 API
```javascript
// 端點: GET /simulations
// 預期輸出格式
{
  "simulations": [
    {
      "id": "SIM-001",
      "name": "批次處理 A",
      "project": "PRJ-001",
      "status": "completed" | "running" | "failed" | "pending",
      "created": "2024-01-15T10:30:00Z",
      "completed": "2024-01-15T11:15:00Z",
      "progress": 100
    }
  ]
}
```

#### 建立模擬 API
```javascript
// 端點: POST /simulations
// 輸入格式
{
  "name": "string",
  "project_id": "string",
  "files": ["file_id"],
  "config": {
    "machine_id": "string",
    "material_id": "string",
    "tool_id": "string"
  }
}

// 預期輸出格式
{
  "simulation_id": "string",
  "status": "created",
  "estimated_duration": 300
}
```

### 事件處理
```javascript
// 輸入事件
@create-new()

// 輸出事件
// 向上傳遞給父組件
```

---

## 9. 迭代選擇模組 (IterationSelection.vue)

### 組件位置
- **文件**: `frontend/src/components/IterationSelection.vue`
- **功能**: 場景選擇和迭代
- **API 依賴**: `/scenarios`, `/scenarios/{id}/iterate`

### UI 元素
- 場景列表卡片
- 狀態指示器
- 選擇按鈕

### API 對應

#### 場景列表 API
```javascript
// 端點: GET /scenarios
// 預期輸出格式
{
  "scenarios": [
    {
      "id": "SCN-001",
      "name": "汽車零件加工專案",
      "project": "PRJ-001",
      "date": "2024-01-15",
      "status": "completed" | "running",
      "version": "1.2",
      "completion": 92
    }
  ]
}
```

#### 迭代場景 API
```javascript
// 端點: POST /scenarios/{id}/iterate
// 輸入格式
{
  "iteration_type": "parameter" | "material" | "tooling" | "strategy",
  "changes": {
    "spindle_speed": 3500,
    "material_id": "new_material_id"
  }
}

// 預期輸出格式
{
  "new_scenario_id": "string",
  "parent_scenario_id": "string",
  "iteration_type": "string",
  "status": "created"
}
```

### 事件處理
```javascript
// 輸入事件
@select-scenario(scenario)

// 輸出事件
// 向上傳遞給父組件
```

---

## 10. 專案選擇模態框 (ProjectSelectionModal.vue)

### 組件位置
- **文件**: `frontend/src/components/ProjectSelectionModal.vue`
- **功能**: 專案和機台選擇
- **API 依賴**: `/projects`, `/machines`

### UI 元素
- 專案下拉選擇
- 機台下拉選擇
- 專案名稱輸入

### API 對應

#### 專案列表 API
```javascript
// 端點: GET /projects
// 預期輸出格式
{
  "projects": [
    {
      "id": "PRJ-001",
      "name": "航太零件 A",
      "description": "string",
      "created": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### 機台列表 API
```javascript
// 端點: GET /machines
// 預期輸出格式
{
  "machines": [
    {
      "id": "cnc-001",
      "name": "Haas VF-2",
      "type": "milling",
      "specs": {
        "max_spindle_speed": 8000,
        "max_feed_rate": 1000
      }
    }
  ]
}
```

### 事件處理
```javascript
// 輸入事件
@close()
@create(data)
// data: { project, machine, name }

// 輸出事件
// 向上傳遞給父組件
```

---

## 11. 迭代類型模態框 (IterationTypeModal.vue)

### 組件位置
- **文件**: `frontend/src/components/IterationTypeModal.vue`
- **功能**: 迭代類型選擇
- **API 依賴**: 無

### UI 元素
- 迭代類型下拉選擇
- 確認/取消按鈕

### 事件處理
```javascript
// 輸入事件
@close()
@start(type)
// type: "parameter" | "material" | "tooling" | "strategy"

// 輸出事件
// 向上傳遞給父組件
```

---

## 狀態管理架構

### 主要狀態 (appStore.js)
```javascript
// 應用程式狀態
currentView: 'landing' | 'login' | 'main'
selectedPlatform: 'v0' | 'v1' | 'data'
activeMode: 'new' | 'iterate' | 'simulation'

// 認證狀態
isAuthenticated: boolean
currentUser: object

// 數據狀態
messages: array
uploadedFiles: array
projects: array
scenarios: array
simulations: array
```

### API 服務層 (apiService.js)
- 統一的 API 請求函數
- 錯誤處理和重試機制
- 請求攔截器 (認證 token)
- 響應攔截器 (錯誤處理)

---

## 錯誤處理

### 常見錯誤響應格式
```javascript
{
  "error": {
    "code": "ERROR_CODE",
    "message": "錯誤描述",
    "details": "詳細信息"
  }
}
```

### 錯誤代碼對應
- `AUTH_REQUIRED`: 需要認證
- `INVALID_CREDENTIALS`: 無效憑證
- `FILE_TOO_LARGE`: 檔案過大
- `INVALID_FILE_FORMAT`: 無效檔案格式
- `OPTIMIZATION_FAILED`: 優化失敗
- `SIMULATION_ERROR`: 模擬錯誤

---

## 開發指南

### 添加新組件
1. 創建 Vue 組件文件
2. 定義 props 和 emits
3. 添加 API 服務函數
4. 更新狀態管理
5. 集成到主應用程式

### API 集成步驟
1. 在 `apiService.js` 中添加新端點
2. 在組件中調用 API 函數
3. 處理響應和錯誤
4. 更新 UI 狀態

### 測試建議
- 組件單元測試
- API 集成測試
- 端到端測試
- 錯誤處理測試 