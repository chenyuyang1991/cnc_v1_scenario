<template>
  <div class="flex-1 p-6">
    <div class="max-w-4xl">
      <div class="mb-8">
        <h1 class="text-2xl font-medium text-white mb-2">迭代專案</h1>
        <p class="text-gray-400">選擇現有專案進行一步的優化和配置調整</p>
      </div>
      
      <div class="mb-6">
        <h2 class="text-lg font-medium text-white mb-4">選擇要迭代的專案</h2>
        <div class="space-y-4">
          <div 
            v-for="scenario in recentScenarios" 
            :key="scenario.id"
            class="bg-black border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
          >
            <div class="flex items-center justify-between">
              <div class="flex-1">
                <div class="flex items-center space-x-3 mb-2">
                  <h3 class="text-white font-medium">{{ scenario.name }} - V{{ scenario.version || '1.2' }}</h3>
                  <span :class="[
                    'px-2 py-1 rounded text-xs font-medium',
                    scenario.status === 'completed' ? 'bg-green-900 text-green-300' :
                    scenario.status === 'running' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-green-900 text-green-300'
                  ]">
                    {{ scenario.status === 'completed' ? '已完成' : '完成中' }}
                  </span>
                </div>
                <div class="flex items-center space-x-6 text-sm text-gray-400">
                  <span>專案 ID: {{ scenario.id }}</span>
                  <span>建立日期: {{ scenario.date }}</span>
                  <span>完成度: {{ scenario.completion || '92' }}%</span>
                </div>
              </div>
              <button 
                @click="$emit('selectScenario', scenario)"
                class="bg-white text-black px-4 py-2 rounded text-sm font-medium hover:bg-gray-100 transition-colors"
              >
                選擇迭代
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  recentScenarios: {
    type: Array,
    required: true
  }
})

defineEmits(['selectScenario'])
</script> 