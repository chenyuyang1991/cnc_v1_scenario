<template>
  <div class="mt-4 border-t border-gray-800 pt-4">
    <div class="bg-gray-900 rounded-lg p-4">
      <h4 class="font-medium text-white mb-3 text-sm">上傳 CNC 程式檔案</h4>
      <div class="space-y-4">
        <div class="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center">
          <Upload class="w-6 h-6 text-gray-400 mx-auto mb-2" />
          <p class="text-gray-400 mb-2 text-sm">拖放檔案至此處，或點擊瀏覽</p>
          <input type="file" multiple class="hidden" ref="fileInput" @change="handleFileUpload" />
          <button 
            @click="$refs.fileInput.click()"
            class="bg-white text-black px-4 py-2 rounded text-sm font-medium hover:bg-gray-100 transition-colors"
          >
            選擇檔案
          </button>
        </div>
        <div class="flex space-x-3">
          <button 
            @click="$emit('download-template', 'excel')"
            class="flex-1 border border-gray-700 text-gray-300 px-4 py-2 rounded text-sm hover:border-gray-600 transition-colors flex items-center justify-center space-x-2"
          >
            <FileSpreadsheet class="w-4 h-4" />
            <span>下載 Excel 範本</span>
          </button>
          <button 
            @click="$emit('download-template', 'csv')"
            class="flex-1 border border-gray-700 text-gray-300 px-4 py-2 rounded text-sm hover:border-gray-600 transition-colors flex items-center justify-center space-x-2"
          >
            <FileText class="w-4 h-4" />
            <span>下載 CSV 範本</span>
          </button>
        </div>
        <div v-if="uploadedFiles.length > 0" class="space-y-2">
          <h5 class="font-medium text-white text-sm">已上傳檔案：</h5>
          <div v-for="file in uploadedFiles" :key="file.name" class="flex items-center justify-between bg-gray-800 rounded p-3">
            <div class="flex items-center space-x-3">
              <FileText class="w-4 h-4 text-gray-400" />
              <span class="text-white text-sm">{{ file.name }}</span>
              <span class="text-gray-400 text-xs">({{ formatFileSize(file.size) }})</span>
            </div>
            <button @click="$emit('file-remove', file)" class="text-gray-400 hover:text-gray-300">
              <X class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
      <div class="flex justify-end space-x-3 mt-4">
        <button class="px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors text-sm">
          取消
        </button>
        <button 
          @click="$emit('validate-files')"
          :disabled="uploadedFiles.length === 0"
          class="px-4 py-2 bg-white text-black rounded text-sm font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          驗證並繼續
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { Upload, FileSpreadsheet, FileText, X } from 'lucide-vue-next'

defineProps({
  uploadedFiles: {
    type: Array,
    default: () => []
  }
})

defineEmits(['file-upload', 'file-remove', 'download-template', 'validate-files'])

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script> 