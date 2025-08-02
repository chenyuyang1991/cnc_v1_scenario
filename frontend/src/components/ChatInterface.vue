<template>
  <div class="flex-1 flex flex-col">
    <!-- Chat Header -->
    <div class="bg-black border-b border-gray-800 p-4">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-lg font-medium text-white">
            {{ getHeaderTitle() }}
          </h1>
          <p class="text-sm text-gray-400">
            {{ getHeaderSubtitle() }}
          </p>
        </div>
        <div class="flex items-center space-x-2">
          <span class="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-800 text-gray-300">
            <div class="w-2 h-2 bg-gray-400 rounded-full mr-2"></div>
            已連線
          </span>
        </div>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      <div v-for="message in messages" :key="message.id" class="flex space-x-3">
        <div v-if="message.type === 'user'" class="flex justify-end w-full">
          <div class="bg-gray-800 text-white rounded-lg px-4 py-2 max-w-md text-sm">
            {{ message.content }}
          </div>
        </div>
        <div v-else class="flex space-x-3 w-full">
          <div class="w-6 h-6 bg-gray-800 rounded flex items-center justify-center flex-shrink-0 mt-1">
            <Bot class="w-3 h-3 text-gray-400" />
          </div>
          <div class="flex-1">
            <div class="bg-black border border-gray-800 rounded-lg p-4">
              <div v-html="message.content" class="text-sm text-gray-300"></div>
              
              <!-- File Upload Section -->
              <FileUploadSection 
                v-if="message.showFileUpload"
                :uploaded-files="uploadedFiles"
                @file-upload="handleFileUpload"
                @file-remove="removeFile"
                @download-template="downloadTemplate"
                @validate-files="validateFiles"
              />

              <!-- Configuration Section -->
              <ConfigurationSection 
                v-if="message.showConfig"
                @close="closeConfig"
                @run-optimization="runOptimization"
              />

              <!-- Results Section -->
              <ResultsSection 
                v-if="message.showResults"
                @close="closeResults"
                @implement-optimization="implementOptimization"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Input -->
    <div class="bg-black border-t border-gray-800 p-4">
      <div class="flex space-x-3">
        <input 
          v-model="chatInput"
          @keypress.enter="sendMessage"
          type="text" 
          placeholder="提出問題或請求修改..."
          class="flex-1 px-3 py-2 bg-black border border-gray-800 rounded text-white placeholder-gray-500 text-sm focus:border-gray-600 focus:outline-none"
        />
        <button 
          @click="sendMessage"
          class="bg-white text-black px-4 py-2 rounded font-medium hover:bg-gray-100 transition-colors flex items-center space-x-2"
        >
          <Send class="w-4 h-4" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { Bot, Send } from 'lucide-vue-next'
import FileUploadSection from './FileUploadSection.vue'
import ConfigurationSection from './ConfigurationSection.vue'
import ResultsSection from './ResultsSection.vue'

const props = defineProps({
  activeMode: {
    type: String,
    required: true
  },
  messages: {
    type: Array,
    required: true
  },
  uploadedFiles: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits([
  'send-message', 
  'file-upload', 
  'file-remove', 
  'download-template', 
  'validate-files',
  'close-config',
  'run-optimization',
  'close-results',
  'implement-optimization'
])

const chatInput = ref('')

const getHeaderTitle = () => {
  if (props.activeMode === 'new') return '新專案生成'
  if (props.activeMode === 'iterate') return '迭代先前專案'
  if (props.activeMode === 'simulation') return '建立模擬'
  return 'CNC 優化助手'
}

const getHeaderSubtitle = () => {
  if (props.activeMode === 'new') return '建立新的優化專案'
  if (props.activeMode === 'iterate') return '修改和改進現有專案'
  if (props.activeMode === 'simulation') return '設定模擬參數'
  return '準備優化您的 CNC 程式'
}

const sendMessage = () => {
  if (!chatInput.value.trim()) return
  emit('send-message', chatInput.value)
  chatInput.value = ''
}

const handleFileUpload = (event) => {
  emit('file-upload', event)
}

const removeFile = (file) => {
  emit('file-remove', file)
}

const downloadTemplate = (type) => {
  emit('download-template', type)
}

const validateFiles = () => {
  emit('validate-files')
}

const closeConfig = () => {
  emit('close-config')
}

const runOptimization = () => {
  emit('run-optimization')
}

const closeResults = () => {
  emit('close-results')
}

const implementOptimization = () => {
  emit('implement-optimization')
}
</script> 