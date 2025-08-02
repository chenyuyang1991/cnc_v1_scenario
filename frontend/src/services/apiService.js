// API 服務層 - 為 FastAPI 後端集成做準備

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// 通用 API 請求函數
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  }

  try {
    const response = await fetch(url, config)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('API request failed:', error)
    throw error
  }
}

// 認證相關 API
export const authAPI = {
  // 登入
  async login(credentials) {
    return apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials)
    })
  },

  // 登出
  async logout() {
    return apiRequest('/auth/logout', {
      method: 'POST'
    })
  },

  // 驗證 token
  async verifyToken() {
    return apiRequest('/auth/verify')
  }
}

// 專案相關 API
export const projectAPI = {
  // 獲取專案列表
  async getProjects() {
    return apiRequest('/projects')
  },

  // 獲取單個專案
  async getProject(id) {
    return apiRequest(`/projects/${id}`)
  },

  // 創建新專案
  async createProject(projectData) {
    return apiRequest('/projects', {
      method: 'POST',
      body: JSON.stringify(projectData)
    })
  },

  // 更新專案
  async updateProject(id, projectData) {
    return apiRequest(`/projects/${id}`, {
      method: 'PUT',
      body: JSON.stringify(projectData)
    })
  },

  // 刪除專案
  async deleteProject(id) {
    return apiRequest(`/projects/${id}`, {
      method: 'DELETE'
    })
  }
}

// 場景相關 API
export const scenarioAPI = {
  // 獲取場景列表
  async getScenarios() {
    return apiRequest('/scenarios')
  },

  // 獲取單個場景
  async getScenario(id) {
    return apiRequest(`/scenarios/${id}`)
  },

  // 創建新場景
  async createScenario(scenarioData) {
    return apiRequest('/scenarios', {
      method: 'POST',
      body: JSON.stringify(scenarioData)
    })
  },

  // 更新場景
  async updateScenario(id, scenarioData) {
    return apiRequest(`/scenarios/${id}`, {
      method: 'PUT',
      body: JSON.stringify(scenarioData)
    })
  },

  // 迭代場景
  async iterateScenario(id, iterationData) {
    return apiRequest(`/scenarios/${id}/iterate`, {
      method: 'POST',
      body: JSON.stringify(iterationData)
    })
  }
}

// 模擬相關 API
export const simulationAPI = {
  // 獲取模擬列表
  async getSimulations() {
    return apiRequest('/simulations')
  },

  // 獲取單個模擬
  async getSimulation(id) {
    return apiRequest(`/simulations/${id}`)
  },

  // 創建新模擬
  async createSimulation(simulationData) {
    return apiRequest('/simulations', {
      method: 'POST',
      body: JSON.stringify(simulationData)
    })
  },

  // 運行模擬
  async runSimulation(id) {
    return apiRequest(`/simulations/${id}/run`, {
      method: 'POST'
    })
  },

  // 獲取模擬結果
  async getSimulationResults(id) {
    return apiRequest(`/simulations/${id}/results`)
  }
}

// 優化相關 API
export const optimizationAPI = {
  // 運行優化
  async runOptimization(optimizationData) {
    return apiRequest('/optimization/run', {
      method: 'POST',
      body: JSON.stringify(optimizationData)
    })
  },

  // 獲取優化結果
  async getOptimizationResults(id) {
    return apiRequest(`/optimization/${id}/results`)
  },

  // 獲取優化配置
  async getOptimizationConfig() {
    return apiRequest('/optimization/config')
  }
}

// 檔案上傳相關 API
export const fileAPI = {
  // 上傳檔案
  async uploadFile(file, type = 'cnc') {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('type', type)

    return apiRequest('/files/upload', {
      method: 'POST',
      headers: {}, // 讓瀏覽器自動設置 Content-Type
      body: formData
    })
  },

  // 驗證檔案
  async validateFile(fileId) {
    return apiRequest(`/files/${fileId}/validate`, {
      method: 'POST'
    })
  },

  // 下載範本
  async downloadTemplate(type) {
    return apiRequest(`/files/templates/${type}`, {
      method: 'GET'
    })
  }
}

// 聊天相關 API
export const chatAPI = {
  // 發送消息
  async sendMessage(message, context = {}) {
    return apiRequest('/chat/send', {
      method: 'POST',
      body: JSON.stringify({ message, context })
    })
  },

  // 獲取聊天歷史
  async getChatHistory(sessionId) {
    return apiRequest(`/chat/history/${sessionId}`)
  }
}

// 配置相關 API
export const configAPI = {
  // 獲取機台配置
  async getMachineConfig() {
    return apiRequest('/config/machine')
  },

  // 獲取材料配置
  async getMaterialConfig() {
    return apiRequest('/config/materials')
  },

  // 獲取刀具配置
  async getToolingConfig() {
    return apiRequest('/config/tooling')
  },

  // 保存配置
  async saveConfig(configType, configData) {
    return apiRequest(`/config/${configType}`, {
      method: 'POST',
      body: JSON.stringify(configData)
    })
  }
}

// 導出所有 API
export default {
  auth: authAPI,
  project: projectAPI,
  scenario: scenarioAPI,
  simulation: simulationAPI,
  optimization: optimizationAPI,
  file: fileAPI,
  chat: chatAPI,
  config: configAPI
} 