# CNC AI 優化器

這是一個完整的 CNC（電腦數值控制）AI 優化平台的前端應用程式。

## 功能特色

- 🚀 三種工作模式：新專案生成、迭代專案、模擬建立
- 🤖 智能聊天界面
- ⚙️ 完整的配置系統（機台、材料、刀具、優化、安全）
- 📊 豐富的結果展示（摘要、圖表、程式碼差異、3D模擬）
- 📁 檔案上傳和驗證系統
- 🎨 現代化黑色主題界面

## 快速開始

### 安裝依賴
```bash
npm install
```

### 開發模式運行
```bash
npm run dev
```

應用程式將在 http://localhost:3000 啟動

### 建置生產版本
```bash
npm run build
```

### 預覽生產版本
```bash
npm run preview
```

## 技術棧

- Vue.js 3 (Composition API)
- Vite
- Tailwind CSS
- Lucide Vue Next (圖標)

## 項目結構

```
frontend/
├── app.vue              # 主要應用組件
├── index.html           # HTML 入口文件
├── package.json         # 項目配置
├── vite.config.js       # Vite 配置
├── tailwind.config.js   # Tailwind 配置
├── postcss.config.js    # PostCSS 配置
└── src/
    ├── main.js          # Vue 應用入口
    └── style.css        # 全局樣式
```

## 使用說明

1. **登陸頁面**: 選擇平台版本（v0/v1/資料上傳）
2. **登入**: 輸入使用者名稱和密碼
3. **主界面**: 選擇工作模式
   - 新專案生成：建立全新的 CNC 優化專案
   - 迭代專案：基於現有專案進行優化
   - 模擬建立：建立和執行 CNC 模擬

## 開發說明

這個應用程式使用 Vue.js 3 的 Composition API 構建，提供了完整的 CNC 優化工作流程界面。所有功能都包含在單個 `app.vue` 文件中，便於理解和修改。 