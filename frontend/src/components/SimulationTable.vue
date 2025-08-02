<template>
  <div class="flex-1 p-6">
    <div class="bg-black border border-gray-800 rounded-lg p-6">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-lg font-medium text-white">模擬執行</h2>
        <div class="flex space-x-3">
          <button 
            @click="$emit('createNew')"
            class="bg-white text-black px-4 py-2 rounded text-sm font-medium hover:bg-gray-100 transition-colors flex items-center space-x-2"
          >
            <Plus class="w-4 h-4" />
            <span>建立新模擬</span>
          </button>
          <button class="border border-gray-800 text-gray-300 px-4 py-2 rounded text-sm hover:border-gray-700 transition-colors">
            匯出資料
          </button>
          <button class="border border-gray-800 text-gray-300 px-4 py-2 rounded text-sm hover:border-gray-700 transition-colors">
            匯入配置
          </button>
        </div>
      </div>
      
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-800">
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">ID</th>
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">名稱</th>
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">專案</th>
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">狀態</th>
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">建立時間</th>
              <th class="text-left py-3 px-4 text-gray-400 font-medium text-sm">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="sim in simulations" :key="sim.id" class="border-b border-gray-800 hover:bg-gray-900">
              <td class="py-3 px-4 text-gray-300 text-sm">{{ sim.id }}</td>
              <td class="py-3 px-4 text-white text-sm">{{ sim.name }}</td>
              <td class="py-3 px-4 text-gray-300 text-sm">{{ sim.project }}</td>
              <td class="py-3 px-4">
                <span :class="[
                  'px-2 py-1 rounded text-xs',
                  sim.status === 'completed' ? 'bg-gray-800 text-gray-300' :
                  sim.status === 'running' ? 'bg-gray-800 text-gray-300' :
                  sim.status === 'failed' ? 'bg-gray-800 text-gray-300' :
                  'bg-gray-800 text-gray-300'
                ]">
                  {{ getStatusText(sim.status) }}
                </span>
              </td>
              <td class="py-3 px-4 text-gray-400 text-sm">{{ sim.created }}</td>
              <td class="py-3 px-4">
                <div class="flex space-x-3">
                  <button class="text-gray-400 hover:text-gray-300 text-sm">檢視</button>
                  <button class="text-gray-400 hover:text-gray-300 text-sm">複製</button>
                  <button class="text-gray-400 hover:text-gray-300 text-sm">刪除</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { Plus } from 'lucide-vue-next'

defineProps({
  simulations: {
    type: Array,
    required: true
  }
})

defineEmits(['createNew'])

const getStatusText = (status) => {
  const statusMap = {
    'completed': '已完成',
    'running': '執行中',
    'failed': '失敗',
    'pending': '待處理'
  }
  return statusMap[status] || status
}
</script> 