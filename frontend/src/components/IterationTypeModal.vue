<template>
  <div v-if="show" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
    <div class="bg-black border border-gray-800 rounded-lg p-6 w-full max-w-md">
      <h3 class="text-lg font-medium text-white mb-4">選擇迭代類型</h3>
      <div class="space-y-4">
        <div>
          <label class="block text-sm text-gray-400 mb-2">迭代類型</label>
          <select v-model="iterationType" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
            <option value="">選擇迭代類型...</option>
            <option value="parameter">參數調整</option>
            <option value="material">材料變更</option>
            <option value="tooling">刀具修改</option>
            <option value="strategy">策略優化</option>
          </select>
        </div>
      </div>
      <div class="flex justify-end space-x-3 mt-6">
        <button 
          @click="$emit('close')"
          class="px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors text-sm"
        >
          取消
        </button>
        <button 
          @click="handleStart"
          :disabled="!iterationType"
          class="px-4 py-2 bg-white text-black rounded text-sm font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          繼續
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  show: {
    type: Boolean,
    required: true
  }
})

const emit = defineEmits(['close', 'start'])

const iterationType = ref('')

const handleStart = () => {
  if (iterationType.value) {
    emit('start', iterationType.value)
    iterationType.value = ''
  }
}

// Reset form when modal closes
watch(() => props.show, (newVal) => {
  if (!newVal) {
    iterationType.value = ''
  }
})
</script> 