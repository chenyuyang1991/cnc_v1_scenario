const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    const token = localStorage.getItem('auth_token')
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers
      },
      ...options
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(error.detail || `HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error)
      throw error
    }
  }

  async login(credentials) {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials)
    })
  }

  async logout() {
    const result = await this.request('/auth/logout', { method: 'POST' })
    localStorage.removeItem('auth_token')
    localStorage.removeItem('user_info')
    return result
  }

  async verifyToken() {
    return this.request('/auth/verify')
  }

  async uploadFile(file) {
    const formData = new FormData()
    formData.append('file', file)
    
    return this.request('/files/upload', {
      method: 'POST',
      headers: {},
      body: formData
    })
  }

  async validateFile(fileId) {
    return this.request(`/files/${fileId}/validate`, { method: 'POST' })
  }

  async downloadTemplate(type) {
    const response = await fetch(`${this.baseURL}/files/templates/${type}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('auth_token')}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Template download failed')
    }
    
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cnc_template.${type === 'excel' ? 'xlsx' : 'csv'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  async getOptimizationConfig() {
    return this.request('/config/optimization')
  }

  async getMachineConfig() {
    return this.request('/config/machine')
  }

  async getMaterialsConfig() {
    return this.request('/config/materials')
  }

  async getToolingConfig() {
    return this.request('/config/tooling')
  }

  async saveConfig(configData) {
    return this.request('/config/save', {
      method: 'POST',
      body: JSON.stringify(configData)
    })
  }

  async runOptimization(config) {
    return this.request('/optimization/run', {
      method: 'POST',
      body: JSON.stringify(config)
    })
  }

  async getOptimizationStatus(optimizationId) {
    return this.request(`/optimization/${optimizationId}/status`)
  }

  async getOptimizationResults(optimizationId) {
    return this.request(`/optimization/${optimizationId}/results`)
  }

  async sendChatMessage(message, context = {}) {
    return this.request('/chat/send', {
      method: 'POST',
      body: JSON.stringify({ message, context })
    })
  }

  async getChatHistory(sessionId) {
    return this.request(`/chat/history/${sessionId}`)
  }

  async getProjects() {
    return {
      projects: [
        { id: 'PRJ-001', name: '航太零件加工', description: '高精度航太零件' },
        { id: 'PRJ-002', name: '汽車零件製造', description: '大批量汽車零件' },
        { id: 'PRJ-003', name: '醫療器材', description: '醫療級精密零件' }
      ]
    }
  }

  async getScenarios() {
    return {
      scenarios: [
        {
          id: 'SCN-001',
          name: '汽車零件加工專案',
          project: 'PRJ-002',
          date: '2024-01-15',
          status: 'completed',
          version: '1.2',
          completion: 92
        },
        {
          id: 'SCN-002', 
          name: '航空零件製造',
          project: 'PRJ-001',
          date: '2024-01-10',
          status: 'running',
          version: '2.1',
          completion: 78
        }
      ]
    }
  }
}

export const apiService = new ApiService()

export const projectAPI = {
  getProjects: () => apiService.getProjects(),
  createProject: (data) => apiService.createProject(data),
  updateProject: (id, data) => apiService.updateProject(id, data),
  deleteProject: (id) => apiService.deleteProject(id)
}

export const configAPI = {
  getOptimizationConfig: () => apiService.getOptimizationConfig(),
  getMachineConfig: () => apiService.getMachineConfig(),
  getMaterialsConfig: () => apiService.getMaterialsConfig(),
  getToolingConfig: () => apiService.getToolingConfig(),
  saveConfig: (data) => apiService.saveConfig(data)
}

export const scenarioAPI = {
  getScenarios: () => apiService.getScenarios(),
  createScenario: (data) => apiService.createScenario(data),
  updateScenario: (id, data) => apiService.updateScenario(id, data),
  deleteScenario: (id) => apiService.deleteScenario(id)
}

export const simulationAPI = {
  runSimulation: (data) => apiService.runSimulation(data),
  getSimulationResults: (id) => apiService.getSimulationResults(id),
  getSimulationStatus: (id) => apiService.getSimulationStatus(id)
}

export const optimizationAPI = {
  runOptimization: (data) => apiService.runOptimization(data),
  getOptimizationResults: (id) => apiService.getOptimizationResults(id),
  getOptimizationStatus: (id) => apiService.getOptimizationStatus(id)
}      