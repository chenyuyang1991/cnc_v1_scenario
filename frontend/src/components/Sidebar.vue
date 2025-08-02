<template>
  <div class="w-64 bg-black border-r border-gray-800 flex flex-col">
    <!-- Header -->
    <div class="p-4 border-b border-gray-800">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="w-8 h-8 bg-white rounded flex items-center justify-center">
            <Settings class="w-4 h-4 text-black" />
          </div>
          <div>
            <h2 class="font-medium text-white text-sm">CNC AI 優化器</h2>
            <p class="text-xs text-gray-400">{{ selectedPlatform.toUpperCase() }}</p>
          </div>
        </div>
        <button @click="$emit('logout')" class="p-1 text-gray-400 hover:text-gray-300">
          <LogOut class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Navigation -->
    <div class="p-4 space-y-1">
      <button 
        @click="$emit('setMode', 'new')"
        :class="[
          'w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center space-x-2',
          activeMode === 'new' ? 'bg-gray-900 text-white' : 'text-gray-400 hover:text-gray-300 hover:bg-gray-900'
        ]"
      >
        <Plus class="w-4 h-4" />
        <span>新專案生成</span>
      </button>
      <button 
        @click="$emit('setMode', 'iterate')"
        :class="[
          'w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center space-x-2',
          activeMode === 'iterate' ? 'bg-gray-900 text-white' : 'text-gray-400 hover:text-gray-300 hover:bg-gray-900'
        ]"
      >
        <RotateCcw class="w-4 h-4" />
        <span>迭代先前專案</span>
      </button>
      <button 
        @click="$emit('setMode', 'simulation')"
        :class="[
          'w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center space-x-2',
          activeMode === 'simulation' ? 'bg-gray-900 text-white' : 'text-gray-400 hover:text-gray-300 hover:bg-gray-900'
        ]"
      >
        <Play class="w-4 h-4" />
        <span>建立模擬</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { Settings, LogOut, Plus, RotateCcw, Play } from 'lucide-vue-next'

defineProps({
  selectedPlatform: {
    type: String,
    required: true
  },
  activeMode: {
    type: String,
    required: true
  }
})

defineEmits(['logout', 'setMode'])
</script> 