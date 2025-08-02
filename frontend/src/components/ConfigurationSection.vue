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
          <!-- 參數設定 Tab -->
          <div v-if="activeConfigTab === 'hyper_params'" class="space-y-6">
            <div class="grid grid-cols-2 gap-6">
              <div v-for="(value, key) in optimizationConfig.hyper_params" :key="key">
                <label class="block text-sm text-gray-400 mb-2">{{ getParamLabel(key) }}</label>
                <input 
                  v-if="typeof value === 'number'"
                  v-model="optimizationConfig.hyper_params[key]"
                  type="number" 
                  :step="getParamStep(key)"
                  class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                />
                <select 
                  v-else-if="typeof value === 'boolean'"
                  v-model="optimizationConfig.hyper_params[key]"
                  class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none"
                >
                  <option :value="true">是</option>
                  <option :value="false">否</option>
                </select>
                <input 
                  v-else
                  v-model="optimizationConfig.hyper_params[key]"
                  type="text" 
                  class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                />
                <p class="text-xs text-gray-500 mt-1">{{ getParamDescription(key) }}</p>
              </div>
            </div>
          </div>
          
          <!-- 子程式設定 Tab -->
          <div v-else-if="activeConfigTab === 'sub_programs'" class="space-y-6">
            <div class="space-y-4">
              <div 
                v-for="(program, programId) in optimizationConfig.sub_programs" 
                :key="programId"
                class="bg-gray-800 rounded-lg p-4"
              >
                <div class="flex items-center justify-between mb-3">
                  <h5 class="text-white font-medium">{{ programId }} - {{ program.function }}</h5>
                  <button 
                    @click="removeSubProgram(programId)"
                    class="text-red-400 hover:text-red-300 text-sm"
                  >
                    刪除
                  </button>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">功能描述</label>
                    <input 
                      v-model="program.function"
                      type="text" 
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                    />
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">刀具</label>
                    <input 
                      v-model="program.tool"
                      type="text" 
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                    />
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">刀具規格</label>
                    <input 
                      v-model="program.tool_spec"
                      type="text" 
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                    />
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">精加工</label>
                    <select 
                      v-model="program.finishing"
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none"
                    >
                      <option :value="0">否</option>
                      <option :value="1">是</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">應用AFC</label>
                    <select 
                      v-model="program.apply_afc"
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none"
                    >
                      <option :value="0">否</option>
                      <option :value="1">是</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">應用Air</label>
                    <select 
                      v-model="program.apply_air"
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none"
                    >
                      <option :value="0">否</option>
                      <option :value="1">是</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">應用Turning</label>
                    <select 
                      v-model="program.apply_turning"
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none"
                    >
                      <option :value="0">否</option>
                      <option :value="1">是</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm text-gray-400 mb-2">最大倍數</label>
                    <input 
                      v-model="program.multiplier_max"
                      type="number" 
                      step="0.1"
                      class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" 
                    />
                  </div>
                </div>
              </div>
              
              <!-- 添加新子程式按鈕 -->
              <button 
                @click="addSubProgram"
                class="w-full p-4 border-2 border-dashed border-gray-700 rounded-lg text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors"
              >
                + 添加新子程式
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div class="flex justify-end space-x-3 mt-6 pt-4 border-t border-gray-800">
        <button @click="$emit('close')" class="px-6 py-2 text-gray-400 hover:text-gray-300 transition-colors">
          取消
        </button>
        <button 
          @click="runOptimization"
          class="px-6 py-2 bg-white text-black rounded font-medium hover:bg-gray-100 transition-colors"
        >
          執行優化
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { X } from 'lucide-vue-next'

const props = defineProps({
  optimizationConfig: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close', 'run-optimization'])

const activeConfigTab = ref('hyper_params')

const configTabs = ref([
  { id: 'hyper_params', name: '參數設定' },
  { id: 'sub_programs', name: '子程式設定' }
])

// 參數標籤映射
const paramLabels = {
  use_cnc_knowledge_base: '使用CNC知識庫',
  percentile_threshold: '百分位閾值',
  short_threshold: '短閾值',
  ae_thres: 'AE閾值',
  ap_thres: 'AP閾值',
  turning_G01_thres: 'Turning G01閾值',
  pre_turning_thres: '預Turning閾值',
  multiplier_max: '最大倍數',
  multiplier_min: '最小倍數',
  multiplier_air: 'Air倍數',
  apply_finishing: '應用精加工',
  apply_ban_n: '應用禁止N',
  multiplier_finishing: '精加工倍數',
  target_pwc_strategy: '目標PWC策略',
  max_increase_step: '最大增加步長',
  min_air_speed: '最小Air速度',
  max_air_speed: '最大Air速度'
}

// 參數描述映射
const paramDescriptions = {
  use_cnc_knowledge_base: '是否使用CNC知識庫進行優化',
  percentile_threshold: '百分位閾值，用於統計分析',
  short_threshold: '短閾值，用於快速判斷',
  ae_thres: 'AE（聲發射）閾值',
  ap_thres: 'AP（進刀深度）閾值',
  turning_G01_thres: 'Turning G01指令閾值',
  pre_turning_thres: '預Turning閾值',
  multiplier_max: '最大倍數限制',
  multiplier_min: '最小倍數限制',
  multiplier_air: 'Air移動倍數',
  apply_finishing: '是否應用精加工策略',
  apply_ban_n: '是否應用禁止N指令',
  multiplier_finishing: '精加工倍數',
  target_pwc_strategy: '目標PWC（功率控制）策略',
  max_increase_step: '最大增加步長',
  min_air_speed: '最小Air移動速度',
  max_air_speed: '最大Air移動速度'
}

// 參數步長映射
const paramSteps = {
  percentile_threshold: 0.01,
  short_threshold: 0.1,
  ae_thres: 0.01,
  ap_thres: 0.01,
  turning_G01_thres: 0.1,
  pre_turning_thres: 0.1,
  multiplier_max: 0.1,
  multiplier_min: 0.1,
  multiplier_air: 0.1,
  apply_finishing: 1,
  apply_ban_n: 1,
  multiplier_finishing: 0.1,
  max_increase_step: 100,
  min_air_speed: 100,
  max_air_speed: 1000
}

const getParamLabel = (key) => {
  return paramLabels[key] || key
}

const getParamDescription = (key) => {
  return paramDescriptions[key] || ''
}

const getParamStep = (key) => {
  return paramSteps[key] || 1
}

const addSubProgram = () => {
  const newId = `5${Math.floor(Math.random() * 900) + 100}`
  props.optimizationConfig.sub_programs[newId] = {
    function: '新子程式',
    tool: '',
    tool_spec: '',
    finishing: 0,
    apply_afc: 1,
    apply_air: 1,
    apply_turning: 1,
    multiplier_max: 1.5,
    ban_n: [],
    ban_row: []
  }
}

const removeSubProgram = (programId) => {
  delete props.optimizationConfig.sub_programs[programId]
}

const runOptimization = () => {
  emit('run-optimization', props.optimizationConfig)
}
</script> 