<template>
  <div class="min-h-screen flex items-center justify-center p-8">
    <div class="max-w-6xl w-full">
      <div class="text-center mb-16">
        <div class="w-20 h-20 bg-white rounded-lg flex items-center justify-center mx-auto mb-6">
          <Settings class="w-10 h-10 text-black" />
        </div>
        <h1 class="text-4xl font-medium text-white mb-4">CNC AI 優化器</h1>
        <p class="text-xl text-gray-400">智能 CNC 程式優化平台</p>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div 
          @click="$emit('navigate', 'login', 'v0')"
          class="bg-black border border-gray-800 rounded-lg p-8 cursor-pointer hover:border-gray-700 transition-all duration-200 group"
        >
          <div class="w-12 h-12 bg-white rounded-md flex items-center justify-center mb-6">
            <Zap class="w-6 h-6 text-black" />
          </div>
          <h3 class="text-xl font-medium text-white mb-4">v0 - 基礎版</h3>
          <p class="text-gray-400 mb-6 text-sm">快速優化專案與標準配置</p>
          <div class="flex items-center text-gray-300 text-sm">
            <span>開始使用</span>
            <ArrowRight class="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>
        
        <div 
          @click="$emit('navigate', 'login', 'v1')"
          class="bg-black border border-gray-800 rounded-lg p-8 cursor-pointer hover:border-gray-700 transition-all duration-200 group"
        >
          <div class="w-12 h-12 bg-white rounded-md flex items-center justify-center mb-6">
            <Cpu class="w-6 h-6 text-black" />
          </div>
          <h3 class="text-xl font-medium text-white mb-4">v1 - 進階版</h3>
          <p class="text-gray-400 mb-6 text-sm">完整功能優化與模擬迭代能力</p>
          <div class="flex items-center text-gray-300 text-sm">
            <span>進入平台</span>
            <ArrowRight class="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>
        
        <div 
          @click="$emit('navigate', 'login', 'data')"
          class="bg-black border border-gray-800 rounded-lg p-8 cursor-pointer hover:border-gray-700 transition-all duration-200 group"
        >
          <div class="w-12 h-12 bg-white rounded-md flex items-center justify-center mb-6">
            <Upload class="w-6 h-6 text-black" />
          </div>
          <h3 class="text-xl font-medium text-white mb-4">資料上傳</h3>
          <p class="text-gray-400 mb-6 text-sm">批量資料處理與批次優化工作流程</p>
          <div class="flex items-center text-gray-300 text-sm">
            <span>上傳資料</span>
            <ArrowRight class="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { Settings, Zap, Cpu, Upload, ArrowRight } from 'lucide-vue-next'

defineEmits(['navigate'])
</script> 