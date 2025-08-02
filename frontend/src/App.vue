<template>
  <div class="min-h-screen bg-black text-white">
    <!-- Master Landing Page -->
    <LandingPage 
      v-if="currentView === 'landing'"
      @navigate="navigateTo"
    />

    <!-- Login Screen -->
    <LoginPage 
      v-else-if="currentView === 'login'"
      :selected-platform="selectedPlatform"
      @login="handleLogin"
      @navigate="navigateTo"
    />

    <!-- Main Application -->
    <div v-else class="min-h-screen flex">
      <!-- Sidebar -->
      <Sidebar 
        :selected-platform="selectedPlatform"
        :active-mode="activeMode"
        @logout="logout"
        @set-mode="setActiveMode"
      />

      <!-- Main Content Area -->
      <div class="flex-1 flex flex-col">
        <!-- Simulation Table View -->
        <SimulationTable 
          v-if="activeMode === 'simulation' && !showSimulationConfig"
          :simulations="simulations"
          @create-new="createNewSimulation"
        />

        <!-- Iterate Scenario Selection View -->
        <IterationSelection 
          v-else-if="activeMode === 'iterate' && !selectedScenarioForIteration"
          :recent-scenarios="recentScenarios"
          @select-scenario="selectScenarioForIteration"
        />

        <!-- Chat Interface for New/Iterate modes or Simulation Config -->
        <ChatInterface 
          v-else
          :active-mode="activeMode"
          :messages="messages"
          :uploaded-files="uploadedFiles"
          @send-message="handleSendMessage"
          @file-upload="handleFileUpload"
          @file-remove="removeFile"
          @download-template="downloadTemplate"
          @validate-files="validateFiles"
          @close-config="closeConfig"
          @run-optimization="runOptimization"
          @close-results="closeResults"
          @implement-optimization="implementOptimization"
        />
      </div>
    </div>

    <!-- Project Selection Modal -->
    <ProjectSelectionModal 
      :show="showProjectModal"
      :projects="projects"
      :machines="machines"
      @close="showProjectModal = false"
      @create="createScenario"
    />

    <!-- Iteration Type Selection Modal -->
    <IterationTypeModal 
      :show="showIterationModal"
      @close="showIterationModal = false"
      @start="startIteration"
    />
  </div>
</template>

<script setup>
import { 
  currentView, selectedPlatform, activeMode, showSimulationConfig, 
  selectedScenarioForIteration, messages, uploadedFiles, showProjectModal, 
  showIterationModal, selectedProject, selectedMachine, scenarioName, 
  iterationType, projects, recentScenarios, simulations, currentScenario,
  machines, initializeData
} from './stores/appStore'

import LandingPage from './components/LandingPage.vue'
import LoginPage from './components/LoginPage.vue'
import Sidebar from './components/Sidebar.vue'
import SimulationTable from './components/SimulationTable.vue'
import IterationSelection from './components/IterationSelection.vue'
import ChatInterface from './components/ChatInterface.vue'
import ProjectSelectionModal from './components/ProjectSelectionModal.vue'
import IterationTypeModal from './components/IterationTypeModal.vue'

// 初始化數據
initializeData()

// Navigation
const navigateTo = (view, platform = '') => {
  currentView.value = view
  if (platform) selectedPlatform.value = platform
}

// Authentication
const handleLogin = (credentials) => {
  if (credentials.username && credentials.password) {
    currentView.value = 'main'
    // TODO: 集成 FastAPI 認證
    console.log('Login attempt:', credentials)
  }
}

const logout = () => {
  currentView.value = 'landing'
  messages.value = []
  currentScenario.value = null
  activeMode.value = ''
  selectedScenarioForIteration.value = null
}

// Mode Management
const setActiveMode = (mode) => {
  activeMode.value = mode
  showSimulationConfig.value = false
  selectedScenarioForIteration.value = null
  
  if (mode === 'new') {
    initializeNewScenario()
  } else if (mode === 'iterate') {
    messages.value = []
  } else if (mode === 'simulation') {
    initializeSimulation()
  }
}

const initializeNewScenario = () => {
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'assistant',
    content: '您好！我準備協助您建立新的優化專案。讓我們先選擇您的專案並配置參數。'
  })
  setTimeout(() => showProjectModal.value = true, 500)
}

const initializeSimulation = () => {
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'assistant',
    content: '歡迎使用模擬建立工作流程。您可以在表格中檢視現有模擬，或上傳 CNC 程式檔案來建立新模擬。'
  })
}

// Scenario Management
const selectScenarioForIteration = (scenario) => {
  selectedScenarioForIteration.value = scenario
  showIterationModal.value = true
}

const startIteration = (type) => {
  showIterationModal.value = false
  currentScenario.value = selectedScenarioForIteration.value
  
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'user',
    content: `迭代專案 ${selectedScenarioForIteration.value.name}，進行 ${type}`
  })
  
  setTimeout(() => {
    messages.value.push({
      id: 2,
      type: 'assistant',
      content: `載入專案「${selectedScenarioForIteration.value.name}」進行迭代。我將向您展示目前的配置，以便您針對${type}進行調整。`,
      showConfig: true
    })
  }, 1000)
  
  iterationType.value = ''
}

const createScenario = (data) => {
  showProjectModal.value = false
  currentScenario.value = {
    name: data.name,
    project: data.project,
    machine: data.machine
  }
  
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: `建立新專案：${data.name}，專案 ${data.project}，機台 ${data.machine}`
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: `完美！我已為專案 ${data.project} 設定專案「${data.name}」。讓我向您展示基於您專案檔案的預設配置選項。`,
      showConfig: true
    })
  }, 1000)
}

// Simulation Management
const createNewSimulation = () => {
  showSimulationConfig.value = true
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '建立新模擬'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '太好了！讓我們先上傳您的 CNC 程式檔案。我已準備了一些範本檔案供您下載，以確保格式正確。',
      showFileUpload: true
    })
  }, 1000)
}

// File Management
const handleFileUpload = (event) => {
  const files = Array.from(event.target.files)
  uploadedFiles.value = [...uploadedFiles.value, ...files]
}

const removeFile = (fileToRemove) => {
  uploadedFiles.value = uploadedFiles.value.filter(file => file !== fileToRemove)
}

const downloadTemplate = (type) => {
  const filename = type === 'excel' ? 'cnc_template.xlsx' : 'cnc_template.csv'
  messages.value.push({
    id: messages.value.length + 1,
    type: 'assistant',
    content: `範本檔案「${filename}」已下載。請填入您的 CNC 程式資料並重新上傳。`
  })
}

const validateFiles = () => {
  if (uploadedFiles.value.length === 0) return
  
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: `驗證 ${uploadedFiles.value.length} 個上傳檔案`
  })
  
  setTimeout(() => {
    const hasErrors = Math.random() > 0.7
    
    if (hasErrors) {
      messages.value.push({
        id: messages.value.length + 1,
        type: 'assistant',
        content: '資料驗證失敗。在上傳檔案中發現問題：<br/>• 檔案 1 缺少必要欄位<br/>• 檔案 2 G-code 語法無效<br/>請修正這些問題並重新上傳。',
        showFileUpload: true
      })
    } else {
      messages.value.push({
        id: messages.value.length + 1,
        type: 'assistant',
        content: '太好了！所有檔案都通過驗證。現在讓我們配置模擬參數。',
        showConfig: true
      })
    }
  }, 2000)
}

// Configuration Management
const closeConfig = () => {
  const configMessage = messages.value.find(m => m.showConfig)
  if (configMessage) {
    configMessage.showConfig = false
  }
}

const runOptimization = () => {
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '使用目前配置執行優化'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '優化完成！這裡是您的結果，包含多個檢視的詳細分析。',
      showResults: true
    })
  }, 2000)
}

// Results Management
const closeResults = () => {
  const resultsMessage = messages.value.find(m => m.showResults)
  if (resultsMessage) {
    resultsMessage.showResults = false
  }
}

const implementOptimization = () => {
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '實施優化變更'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '太棒了！優化的 CNC 程式已成功實施。變更已儲存到您的專案中。您現在可以執行此專案或基於這些結果建立變化版本。'
    })
    
    if (activeMode.value === 'simulation') {
      simulations.value.unshift({
        id: `SIM-${String(simulations.value.length + 1).padStart(3, '0')}`,
        name: currentScenario.value?.name || '新模擬',
        project: currentScenario.value?.project || 'X1111-CNC2',
        status: 'completed',
        created: new Date().toISOString().split('T')[0]
      })
      
      setTimeout(() => {
        showSimulationConfig.value = false
        messages.value.push({
          id: messages.value.length + 1,
          type: 'assistant',
          content: '模擬已新增到您的模擬執行表格中。您可以在主模擬儀表板中檢視。'
        })
      }, 1000)
    }
  }, 1000)
}

// Chat Management
const handleSendMessage = (message) => {
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: message
  })
  
  const userMessage = message.toLowerCase()
  
  setTimeout(() => {
    let response = '我了解您的要求。'
    
    if (userMessage.includes('修改') || userMessage.includes('變更')) {
      response += '我可以協助您修改目前的配置。您想要調整哪些特定參數？'
    } else if (userMessage.includes('解釋') || userMessage.includes('為什麼')) {
      response += '讓我詳細解釋優化結果。改進來自於優化的進給速度、更好的刀路策略和減少非切削時間。'
    } else if (userMessage.includes('比較')) {
      response += '我可以向您展示原始和優化程式之間的詳細比較。您想要查看效能指標還是程式碼差異？'
    } else if (userMessage.includes('模擬')) {
      response += '對於模擬相關問題，我可以協助您了解 3D 視覺化、材料移除分析或碰撞檢測結果。'
    } else {
      response += '我還能如何協助您進行 CNC 優化？我可以協助參數調整、結果分析或建立新專案。'
    }
    
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: response
    })
  }, 1000)
}
</script> 