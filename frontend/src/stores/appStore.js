import { ref, reactive } from 'vue'
import { projectAPI, configAPI, scenarioAPI, simulationAPI } from '../services/apiService'

// 應用程式狀態
export const currentView = ref('landing')
export const selectedPlatform = ref('')
export const activeMode = ref('')
export const showSimulationConfig = ref(false)
export const selectedScenarioForIteration = ref(null)

// 認證狀態
export const isAuthenticated = ref(false)
export const currentUser = ref(null)

// 聊天和消息
export const messages = ref([])
export const chatInput = ref('')

// 檔案上傳
export const uploadedFiles = ref([])

// 模態框狀態
export const showProjectModal = ref(false)
export const showIterationModal = ref(false)

// 表單數據
export const selectedProject = ref('')
export const selectedMachine = ref('')
export const scenarioName = ref('')
export const iterationType = ref('')

// 標籤頁狀態
export const activeConfigTab = ref('machine')
export const activeResultTab = ref('summary')

// 動態數據 - 從API獲取
export const projects = ref([])
export const recentScenarios = ref([])
export const simulations = ref([])
export const machines = ref([])

// 加載狀態
export const isLoading = ref(false)
export const error = ref(null)

// 配置標籤頁
export const configTabs = ref([
  { id: 'machine', name: '機台' },
  { id: 'material', name: '材料' },
  { id: 'tooling', name: '刀具' },
  { id: 'optimization', name: '優化' },
  { id: 'safety', name: '安全' }
])

// 結果標籤頁
export const resultTabs = ref([
  { id: 'summary', name: '摘要' },
  { id: 'charts', name: '圖表' },
  { id: 'code', name: '程式碼差異' },
  { id: 'simulation', name: '模擬' },
  { id: 'validation', name: '驗證' }
])

// 當前場景
export const currentScenario = ref(null)

// API 數據加載函數
export const loadProjects = async () => {
  try {
    isLoading.value = true
    error.value = null
    const response = await projectAPI.getProjects()
    projects.value = response.projects || []
  } catch (error) {
    console.error('Failed to load projects:', error)
    error.value = 'Failed to load projects'
    projects.value = []
  } finally {
    isLoading.value = false
  }
}

export const loadMachines = async () => {
  try {
    isLoading.value = true
    error.value = null
    const response = await configAPI.getMachineConfig()
    machines.value = response.machines || []
  } catch (error) {
    console.error('Failed to load machines:', error)
    error.value = 'Failed to load machines'
    machines.value = []
  } finally {
    isLoading.value = false
  }
}

export const loadScenarios = async () => {
  try {
    isLoading.value = true
    error.value = null
    const response = await scenarioAPI.getScenarios()
    recentScenarios.value = response.scenarios || []
  } catch (error) {
    console.error('Failed to load scenarios:', error)
    error.value = 'Failed to load scenarios'
    recentScenarios.value = []
  } finally {
    isLoading.value = false
  }
}

export const loadSimulations = async () => {
  try {
    isLoading.value = true
    error.value = null
    const response = await simulationAPI.getSimulations()
    simulations.value = response.simulations || []
  } catch (error) {
    console.error('Failed to load simulations:', error)
    error.value = 'Failed to load simulations'
    simulations.value = []
  } finally {
    isLoading.value = false
  }
}

// 初始化數據
export const initializeData = async () => {
  try {
    isLoading.value = true
    error.value = null
    await Promise.all([
      loadProjects(),
      loadMachines(),
      loadScenarios(),
      loadSimulations()
    ])
  } catch (error) {
    console.error('Failed to initialize data:', error)
    error.value = 'Failed to initialize data'
  } finally {
    isLoading.value = false
  }
} 