<template>
  <div v-if="show" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
    <div class="bg-black border border-gray-800 rounded-lg p-6 w-full max-w-md">
      <h3 class="text-lg font-medium text-white mb-4">選擇專案</h3>
      <div class="space-y-4">
        <div>
          <label class="block text-sm text-gray-400 mb-2">專案 ID</label>
          <select v-model="selectedProject" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
            <option value="">選擇專案...</option>
            <option v-for="project in projects" :key="project.id" :value="project.id">
              {{ project.name }} ({{ project.id }})
            </option>
          </select>
        </div>
        <div v-if="selectedProject">
          <label class="block text-sm text-gray-400 mb-2">機台</label>
          <select v-model="selectedMachine" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
            <option value="">選擇機台...</option>
            <option value="cnc-001">CNC-001 (Haas VF-2)</option>
            <option value="cnc-002">CNC-002 (Mazak VTC-200)</option>
            <option value="cnc-003">CNC-003 (DMG Mori NHX-4000)</option>
          </select>
        </div>
        <div v-if="selectedMachine">
          <label class="block text-sm text-gray-400 mb-2">專案名稱</label>
          <input 
            v-model="scenarioName"
            type="text" 
            placeholder="輸入專案名稱..."
            class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white placeholder-gray-500 text-sm focus:border-gray-600 focus:outline-none"
          />
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
          @click="handleCreate"
          :disabled="!selectedProject || !selectedMachine || !scenarioName"
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
  },
  projects: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['close', 'create'])

const selectedProject = ref('')
const selectedMachine = ref('')
const scenarioName = ref('')

const handleCreate = () => {
  if (selectedProject.value && selectedMachine.value && scenarioName.value) {
    emit('create', {
      project: selectedProject.value,
      machine: selectedMachine.value,
      name: scenarioName.value
    })
    
    // Reset form
    selectedProject.value = ''
    selectedMachine.value = ''
    scenarioName.value = ''
  }
}

// Reset form when modal closes
watch(() => props.show, (newVal) => {
  if (!newVal) {
    selectedProject.value = ''
    selectedMachine.value = ''
    scenarioName.value = ''
  }
})
</script> 