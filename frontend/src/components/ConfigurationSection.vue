<template>
  <div class="mt-4 border-t border-gray-800 pt-4">
    <div class="bg-gray-900 rounded-lg p-4">
      <div class="flex items-center justify-between mb-4">
        <h4 class="font-medium text-white text-lg">配置設定</h4>
        <button @click="$emit('close')" class="text-gray-400 hover:text-gray-300">
          <X class="w-5 h-5" />
        </button>
      </div>
      
      <div class="flex">
        <!-- Tab Navigation -->
        <div class="w-48 border-r border-gray-800 pr-4">
          <div class="space-y-1">
            <button 
              v-for="tab in configTabs" 
              :key="tab.id"
              @click="activeConfigTab = tab.id"
              :class="[
                'w-full text-left px-3 py-2 text-sm rounded transition-colors',
                activeConfigTab === tab.id 
                  ? 'bg-white text-black' 
                  : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800'
              ]"
            >
              {{ tab.name }}
            </button>
          </div>
        </div>
        
        <!-- Tab Content -->
        <div class="flex-1 pl-6">
          <div v-if="activeConfigTab === 'machine'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div>
                <label class="block text-sm text-gray-400 mb-2">主軸轉速 (RPM)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="3000" />
                <p class="text-xs text-gray-500 mt-1">建議：2000-5000 RPM</p>
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-2">進給速度 (mm/min)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="500" />
                <p class="text-xs text-gray-500 mt-1">建議：300-800 mm/min</p>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeConfigTab === 'material'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div>
                <label class="block text-sm text-gray-400 mb-2">材料類型</label>
                <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                  <option>鋁合金 6061-T6</option>
                  <option>碳鋼 1018</option>
                  <option>不鏽鋼 304</option>
                </select>
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-2">硬度 (HRC)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="25" />
              </div>
            </div>
          </div>
          
          <div v-else-if="activeConfigTab === 'tooling'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div>
                <label class="block text-sm text-gray-400 mb-2">刀具類型</label>
                <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                  <option>端銑刀 - 平底</option>
                  <option>端銑刀 - 球頭</option>
                  <option>面銑刀</option>
                </select>
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-2">刀具直徑 (mm)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="6" />
              </div>
            </div>
          </div>
          
          <div v-else-if="activeConfigTab === 'optimization'" class="space-y-6">
            <div class="bg-gray-800 rounded-lg p-4">
              <h5 class="text-white font-medium mb-3">優化目標</h5>
              <div class="space-y-4">
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-3">
                    <input type="checkbox" id="time-opt" class="bg-black border-gray-800" checked />
                    <label for="time-opt" class="text-sm text-gray-300">最小化加工時間</label>
                  </div>
                  <div class="text-xs text-gray-500">優先級：高</div>
                </div>
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-3">
                    <input type="checkbox" id="quality-opt" class="bg-black border-gray-800" />
                    <label for="quality-opt" class="text-sm text-gray-300">優化表面品質</label>
                  </div>
                  <div class="text-xs text-gray-500">優先級：中</div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeConfigTab === 'safety'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div>
                <label class="block text-sm text-gray-400 mb-2">最大主軸轉速 (RPM)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="5000" />
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-2">最大進給速度 (mm/min)</label>
                <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="1000" />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="flex justify-end space-x-3 mt-6 pt-4 border-t border-gray-800">
        <button @click="$emit('close')" class="px-6 py-2 text-gray-400 hover:text-gray-300 transition-colors">
          取消
        </button>
        <button 
          @click="$emit('run-optimization')"
          class="px-6 py-2 bg-white text-black rounded font-medium hover:bg-gray-100 transition-colors"
        >
          執行優化
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { X } from 'lucide-vue-next'

defineEmits(['close', 'run-optimization'])

const activeConfigTab = ref('machine')

const configTabs = ref([
  { id: 'machine', name: '機台' },
  { id: 'material', name: '材料' },
  { id: 'tooling', name: '刀具' },
  { id: 'optimization', name: '優化' },
  { id: 'safety', name: '安全' }
])
</script> 