<template>
  <div class="mt-4 border-t border-gray-800 pt-4">
    <div class="bg-gray-900 rounded-lg p-4">
      <div class="flex items-center justify-between mb-4">
        <h4 class="font-medium text-white text-lg">優化結果</h4>
        <button @click="$emit('close')" class="text-gray-400 hover:text-gray-300">
          <X class="w-5 h-5" />
        </button>
      </div>
      
      <div class="flex">
        <!-- Tab Navigation -->
        <div class="w-48 border-r border-gray-800 pr-4">
          <div class="space-y-1">
            <button 
              v-for="tab in resultTabs" 
              :key="tab.id"
              @click="activeResultTab = tab.id"
              :class="[
                'w-full text-left px-3 py-2 text-sm rounded transition-colors',
                activeResultTab === tab.id 
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
          <div v-if="activeResultTab === 'summary'" class="space-y-6">
            <div class="grid grid-cols-4 gap-4 mb-6">
              <div class="text-center p-4 bg-gray-800 rounded-lg">
                <div class="text-2xl font-medium text-white">-23%</div>
                <div class="text-sm text-gray-400">加工時間</div>
                <div class="text-xs text-gray-500 mt-1">節省 10.4 分鐘</div>
              </div>
              <div class="text-center p-4 bg-gray-800 rounded-lg">
                <div class="text-2xl font-medium text-white">+15%</div>
                <div class="text-sm text-gray-400">刀具壽命</div>
                <div class="text-xs text-gray-500 mt-1">約多 150 件</div>
              </div>
              <div class="text-center p-4 bg-gray-800 rounded-lg">
                <div class="text-2xl font-medium text-white">98.5%</div>
                <div class="text-sm text-gray-400">品質分數</div>
                <div class="text-xs text-gray-500 mt-1">表面粗糙度：Ra 0.8</div>
              </div>
              <div class="text-center p-4 bg-gray-800 rounded-lg">
                <div class="text-2xl font-medium text-white">$127</div>
                <div class="text-sm text-gray-400">成本節省</div>
                <div class="text-xs text-gray-500 mt-1">每件</div>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeResultTab === 'charts'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div class="bg-gray-800 rounded-lg p-4">
                <h5 class="text-white font-medium mb-3">加工時間比較</h5>
                <div class="h-64 bg-gray-700 rounded flex items-center justify-center">
                  <div class="text-center">
                    <BarChart3 class="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p class="text-gray-400 text-sm">互動式長條圖</p>
                    <p class="text-gray-500 text-xs">原始 vs 優化</p>
                  </div>
                </div>
              </div>
              
              <div class="bg-gray-800 rounded-lg p-4">
                <h5 class="text-white font-medium mb-3">刀具磨損分析</h5>
                <div class="h-64 bg-gray-700 rounded flex items-center justify-center">
                  <div class="text-center">
                    <TrendingUp class="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p class="text-gray-400 text-sm">刀具磨損進程</p>
                    <p class="text-gray-500 text-xs">時間/件數</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeResultTab === 'code'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6 h-96">
              <div class="bg-gray-800 rounded-lg p-4 flex flex-col">
                <h5 class="text-white font-medium mb-3">原始 G-Code</h5>
                <div class="flex-1 bg-black text-gray-300 p-4 rounded text-sm font-mono overflow-y-auto">
                  <div class="space-y-1">
                    <div>G90 G54 G17 G49 G40 G80</div>
                    <div>T1 M6</div>
                    <div>G43 H1 Z25.</div>
                    <div>S3000 M3</div>
                    <div>G00 X0 Y0</div>
                  </div>
                </div>
              </div>
              
              <div class="bg-gray-800 rounded-lg p-4 flex flex-col">
                <h5 class="text-white font-medium mb-3">優化後 G-Code</h5>
                <div class="flex-1 bg-black text-gray-300 p-4 rounded text-sm font-mono overflow-y-auto">
                  <div class="space-y-1">
                    <div>G90 G54 G17 G49 G40 G80</div>
                    <div>T1 M6</div>
                    <div>G43 H1 Z25.</div>
                    <div class="bg-gray-700">S3500 M3</div>
                    <div>G00 X0 Y0</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeResultTab === 'simulation'" class="space-y-6">
            <div class="bg-gray-800 rounded-lg p-4">
              <h5 class="text-white font-medium mb-3">3D 加工模擬</h5>
              <div class="h-96 bg-gray-700 rounded flex items-center justify-center">
                <div class="text-center">
                  <Box class="w-20 h-20 text-gray-400 mx-auto mb-4" />
                  <p class="text-gray-400 mb-4 text-lg">3D CNC 模擬檢視器</p>
                  <div class="space-y-2">
                    <button class="px-6 py-2 bg-white text-black rounded font-medium mr-3">
                      播放模擬
                    </button>
                    <button class="px-6 py-2 border border-gray-600 text-gray-300 rounded">
                      逐步執行
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div v-else-if="activeResultTab === 'validation'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div class="bg-gray-800 rounded-lg p-4">
                <h5 class="text-white font-medium mb-3">安全驗證</h5>
                <div class="space-y-3">
                  <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                    <CheckCircle class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-300 text-sm">所有安全檢查通過</span>
                  </div>
                  <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                    <CheckCircle class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-300 text-sm">刀具碰撞分析：清除</span>
                  </div>
                </div>
              </div>
              
              <div class="bg-gray-800 rounded-lg p-4">
                <h5 class="text-white font-medium mb-3">品質驗證</h5>
                <div class="space-y-3">
                  <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                    <CheckCircle class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-300 text-sm">表面光潔度在公差範圍內</span>
                  </div>
                  <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                    <CheckCircle class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-300 text-sm">尺寸精度：±0.02mm</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="flex justify-end space-x-3 mt-6 pt-4 border-t border-gray-800">
        <button @click="$emit('close')" class="px-6 py-2 text-gray-400 hover:text-gray-300 transition-colors">
          關閉
        </button>
        <button class="px-6 py-2 border border-gray-700 text-gray-300 rounded hover:border-gray-600 transition-colors">
          匯出結果
        </button>
        <button 
          @click="$emit('implement-optimization')"
          class="px-6 py-2 bg-white text-black rounded font-medium hover:bg-gray-100 transition-colors"
        >
          實施變更
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { X, BarChart3, TrendingUp, Box, CheckCircle } from 'lucide-vue-next'

defineEmits(['close', 'implement-optimization'])

const activeResultTab = ref('summary')

const resultTabs = ref([
  { id: 'summary', name: '摘要' },
  { id: 'charts', name: '圖表' },
  { id: 'code', name: '程式碼差異' },
  { id: 'simulation', name: '模擬' },
  { id: 'validation', name: '驗證' }
])
</script> 