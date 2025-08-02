import { ref, reactive } from 'vue'

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

// 靜態數據
export const projects = ref([
  { id: 'PRJ-001', name: '航太零件 A' },
  { id: 'PRJ-002', name: '汽車零件 B' },
  { id: 'PRJ-003', name: '醫療器材 C' }
])

export const recentScenarios = ref([
  { id: 'SCN-001', name: '汽車零件加工專案', project: 'PRJ-001', date: '2024-01-15', type: '專案', status: 'completed', version: '1.2', completion: '92' },
  { id: 'SCN-002', name: '航空零件製造專案', project: 'PRJ-002', date: '2024-01-14', type: '專案', status: 'running', version: '2.1', completion: '88' },
  { id: 'SCN-003', name: '精密模具專案', project: 'PRJ-001', date: '2024-01-13', type: '專案', status: 'completed', version: '1.0', completion: '95' }
])

export const simulations = ref([
  { id: 'SIM-001', name: '批次處理 A', project: 'PRJ-001', status: 'completed', created: '2024-01-15' },
  { id: 'SIM-002', name: '品質測試 B', project: 'PRJ-002', status: 'running', created: '2024-01-14' },
  { id: 'SIM-003', name: '效能測試 C', project: 'PRJ-003', status: 'failed', created: '2024-01-13' },
  { id: 'SIM-004', name: '負載測試 D', project: 'PRJ-001', status: 'pending', created: '2024-01-12' }
])

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