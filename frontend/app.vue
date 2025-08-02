<template>
  <div class="min-h-screen bg-black text-white">
    <!-- Master Landing Page -->
    <div v-if="currentView === 'landing'" class="min-h-screen flex items-center justify-center p-8">
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
            @click="navigateTo('login', 'v0')"
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
            @click="navigateTo('login', 'v1')"
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
            @click="navigateTo('login', 'data')"
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

    <!-- Login Screen -->
    <div v-else-if="currentView === 'login'" class="min-h-screen flex items-center justify-center p-4">
      <div class="bg-black border border-gray-800 rounded-lg p-8 w-full max-w-md">
        <div class="text-center mb-8">
          <div class="w-12 h-12 bg-white rounded-md flex items-center justify-center mx-auto mb-4">
            <Settings class="w-6 h-6 text-black" />
          </div>
          <h1 class="text-xl font-medium text-white">登入</h1>
          <p class="text-gray-400 mt-2 text-sm">存取 {{ selectedPlatform.toUpperCase() }} 平台</p>
        </div>
        
        <form @submit.prevent="login" class="space-y-4">
          <div>
            <label class="block text-sm text-gray-300 mb-2">使用者名稱</label>
            <input 
              v-model="loginForm.username"
              type="text" 
              class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white placeholder-gray-500 focus:border-gray-600 focus:outline-none"
              placeholder="輸入您的使用者名稱"
            />
          </div>
          <div>
            <label class="block text-sm text-gray-300 mb-2">密碼</label>
            <input 
              v-model="loginForm.password"
              type="password" 
              class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white placeholder-gray-500 focus:border-gray-600 focus:outline-none"
              placeholder="輸入您的密碼"
            />
          </div>
          <button 
            type="submit"
            class="w-full bg-white text-black py-2 rounded font-medium hover:bg-gray-100 transition-colors"
          >
            登入
          </button>
        </form>
        
        <button 
          @click="currentView = 'landing'"
          class="w-full mt-4 text-gray-400 hover:text-gray-300 transition-colors text-sm"
        >
          ← 返回平台選擇
        </button>
      </div>
    </div>

    <!-- Main Application -->
    <div v-else class="min-h-screen flex">
      <!-- Sidebar -->
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
            <button @click="logout" class="p-1 text-gray-400 hover:text-gray-300">
              <LogOut class="w-4 h-4" />
            </button>
          </div>
        </div>

        <!-- Navigation -->
        <div class="p-4 space-y-1">
          <button 
            @click="setActiveMode('new')"
            :class="[
              'w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center space-x-2',
              activeMode === 'new' ? 'bg-gray-900 text-white' : 'text-gray-400 hover:text-gray-300 hover:bg-gray-900'
            ]"
          >
            <Plus class="w-4 h-4" />
            <span>新專案生成</span>
          </button>
          <button 
            @click="setActiveMode('iterate')"
            :class="[
              'w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center space-x-2',
              activeMode === 'iterate' ? 'bg-gray-900 text-white' : 'text-gray-400 hover:text-gray-300 hover:bg-gray-900'
            ]"
          >
            <RotateCcw class="w-4 h-4" />
            <span>迭代先前專案</span>
          </button>
          <button 
            @click="setActiveMode('simulation')"
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

      <!-- Main Content Area -->
      <div class="flex-1 flex flex-col">
        <!-- Simulation Table View -->
        <div v-if="activeMode === 'simulation' && !showSimulationConfig" class="flex-1 p-6">
          <div class="bg-black border border-gray-800 rounded-lg p-6">
            <div class="flex items-center justify-between mb-6">
              <h2 class="text-lg font-medium text-white">模擬執行</h2>
              <div class="flex space-x-3">
                <button 
                  @click="createNewSimulation"
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

        <!-- Iterate Scenario Selection View -->
        <div v-else-if="activeMode === 'iterate' && !selectedScenarioForIteration" class="flex-1 p-6">
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
                      @click="selectScenarioForIteration(scenario)"
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

        <!-- Chat Interface for New/Iterate modes or Simulation Config -->
        <div v-else class="flex-1 flex flex-col">
          <!-- Chat Header -->
          <div class="bg-black border-b border-gray-800 p-4">
            <div class="flex items-center justify-between">
              <div>
                <h1 class="text-lg font-medium text-white">
                  {{ getHeaderTitle() }}
                </h1>
                <p class="text-sm text-gray-400">
                  {{ getHeaderSubtitle() }}
                </p>
              </div>
              <div class="flex items-center space-x-2">
                <span class="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-800 text-gray-300">
                  <div class="w-2 h-2 bg-gray-400 rounded-full mr-2"></div>
                  已連線
                </span>
              </div>
            </div>
          </div>

          <!-- Chat Messages -->
          <div class="flex-1 overflow-y-auto p-4 space-y-4">
            <div v-for="message in messages" :key="message.id" class="flex space-x-3">
              <div v-if="message.type === 'user'" class="flex justify-end w-full">
                <div class="bg-gray-800 text-white rounded-lg px-4 py-2 max-w-md text-sm">
                  {{ message.content }}
                </div>
              </div>
              <div v-else class="flex space-x-3 w-full">
                <div class="w-6 h-6 bg-gray-800 rounded flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot class="w-3 h-3 text-gray-400" />
                </div>
                <div class="flex-1">
                  <div class="bg-black border border-gray-800 rounded-lg p-4">
                    <div v-html="message.content" class="text-sm text-gray-300"></div>
                    
                    <!-- File Upload Section -->
                    <div v-if="message.showFileUpload" class="mt-4 border-t border-gray-800 pt-4">
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
                              @click="downloadTemplate('excel')"
                              class="flex-1 border border-gray-700 text-gray-300 px-4 py-2 rounded text-sm hover:border-gray-600 transition-colors flex items-center justify-center space-x-2"
                            >
                              <FileSpreadsheet class="w-4 h-4" />
                              <span>下載 Excel 範本</span>
                            </button>
                            <button 
                              @click="downloadTemplate('csv')"
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
                              <button @click="removeFile(file)" class="text-gray-400 hover:text-gray-300">
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
                            @click="validateFiles"
                            :disabled="uploadedFiles.length === 0"
                            class="px-4 py-2 bg-white text-black rounded text-sm font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            驗證並繼續
                          </button>
                        </div>
                      </div>
                    </div>

                    <!-- Configuration Section - Full Chat Width -->
                    <div v-if="message.showConfig" class="mt-4 border-t border-gray-800 pt-4">
                      <div class="bg-gray-900 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-4">
                          <h4 class="font-medium text-white text-lg">配置設定</h4>
                          <button @click="closeConfig" class="text-gray-400 hover:text-gray-300">
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
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">下刀速度 (mm/min)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="200" />
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">冷卻液類型</label>
                                  <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                    <option>水溶性冷卻液</option>
                                    <option>霧狀冷卻液</option>
                                    <option>氣吹</option>
                                    <option>無</option>
                                  </select>
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">機台限制</h5>
                                <div class="grid grid-cols-3 gap-4">
                                  <div>
                                    <label class="block text-xs text-gray-400 mb-1">X軸最大行程</label>
                                    <input type="number" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm" value="500" />
                                  </div>
                                  <div>
                                    <label class="block text-xs text-gray-400 mb-1">Y軸最大行程</label>
                                    <input type="number" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm" value="400" />
                                  </div>
                                  <div>
                                    <label class="block text-xs text-gray-400 mb-1">Z軸最大行程</label>
                                    <input type="number" class="w-full px-3 py-2 bg-black border border-gray-800 rounded text-white text-sm" value="300" />
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <div v-else-if="activeConfigTab === 'material'" class="space-y-6">
                              <div class="grid grid-cols-2 gap-6">
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">材料類型</label>
                                  <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                    <option>鋁合金 6061-T6</option>
                                    <option>鋁合金 7075-T6</option>
                                    <option>碳鋼 1018</option>
                                    <option>合金鋼 4140</option>
                                    <option>不鏽鋼 304</option>
                                    <option>不鏽鋼 316</option>
                                    <option>鈦合金 Ti-6Al-4V</option>
                                  </select>
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">硬度 (HRC)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="25" />
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">料厚 (mm)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="10" />
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">料寬 (mm)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="100" />
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">材料特性</h5>
                                <div class="grid grid-cols-2 gap-4">
                                  <div>
                                    <span class="text-sm text-gray-400">抗拉強度：</span>
                                    <span class="text-white ml-2">310 MPa</span>
                                  </div>
                                  <div>
                                    <span class="text-sm text-gray-400">降伏強度：</span>
                                    <span class="text-white ml-2">276 MPa</span>
                                  </div>
                                  <div>
                                    <span class="text-sm text-gray-400">密度：</span>
                                    <span class="text-white ml-2">2.70 g/cm³</span>
                                  </div>
                                  <div>
                                    <span class="text-sm text-gray-400">熱傳導率：</span>
                                    <span class="text-white ml-2">167 W/m·K</span>
                                  </div>
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
                                    <option>端銑刀 - 圓角</option>
                                    <option>面銑刀</option>
                                    <option>鑽頭</option>
                                    <option>鉸刀</option>
                                    <option>攻牙器</option>
                                  </select>
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">刀具直徑 (mm)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="6" />
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">刃數</label>
                                  <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                    <option>2</option>
                                    <option>3</option>
                                    <option>4</option>
                                    <option>6</option>
                                  </select>
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">刀具長度 (mm)</label>
                                  <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="50" />
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">刀路策略</h5>
                                <div class="space-y-3">
                                  <div class="flex items-center space-x-3">
                                    <input type="radio" id="conventional" name="strategy" class="bg-black border-gray-800" checked />
                                    <label for="conventional" class="text-sm text-gray-300">傳統銑削</label>
                                  </div>
                                  <div class="flex items-center space-x-3">
                                    <input type="radio" id="climb" name="strategy" class="bg-black border-gray-800" />
                                    <label for="climb" class="text-sm text-gray-300">順銑</label>
                                  </div>
                                  <div class="flex items-center space-x-3">
                                    <input type="radio" id="adaptive" name="strategy" class="bg-black border-gray-800" />
                                    <label for="adaptive" class="text-sm text-gray-300">適應性清角</label>
                                  </div>
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
                                  <div class="flex items-center justify-between">
                                    <div class="flex items-center space-x-3">
                                      <input type="checkbox" id="tool-life" class="bg-black border-gray-800" checked />
                                      <label for="tool-life" class="text-sm text-gray-300">最大化刀具壽命</label>
                                    </div>
                                    <div class="text-xs text-gray-500">優先級：高</div>
                                  </div>
                                  <div class="flex items-center justify-between">
                                    <div class="flex items-center space-x-3">
                                      <input type="checkbox" id="power-opt" class="bg-black border-gray-800" />
                                      <label for="power-opt" class="text-sm text-gray-300">最小化功耗</label>
                                    </div>
                                    <div class="text-xs text-gray-500">優先級：低</div>
                                  </div>
                                  
                                  <div class="grid grid-cols-2 gap-6 mt-6">
                                    <div>
                                      <label class="block text-sm text-gray-400 mb-2">優化演算法</label>
                                      <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                        <option>遺傳演算法</option>
                                        <option>粒子群優化</option>
                                        <option>模擬退火</option>
                                        <option>梯度下降</option>
                                      </select>
                                    </div>
                                    <div>
                                      <label class="block text-sm text-gray-400 mb-2">迭代次數</label>
                                      <input type="number" class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none" value="1000" />
                                    </div>
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
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">安全係數</label>
                                  <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                    <option>保守 (0.7)</option>
                                    <option>標準 (0.8)</option>
                                    <option>積極 (0.9)</option>
                                  </select>
                                </div>
                                <div>
                                  <label class="block text-sm text-gray-400 mb-2">碰撞檢測</label>
                                  <select class="w-full px-4 py-3 bg-black border border-gray-800 rounded text-white text-sm focus:border-gray-600 focus:outline-none">
                                    <option>啟用</option>
                                    <option>停用</option>
                                  </select>
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">安全檢查</h5>
                                <div class="space-y-3">
                                  <div class="flex items-center space-x-3">
                                    <input type="checkbox" id="rapid-check" class="bg-black border-gray-800" checked />
                                    <label for="rapid-check" class="text-sm text-gray-300">快速移動碰撞檢查</label>
                                  </div>
                                  <div class="flex items-center space-x-3">
                                    <input type="checkbox" id="spindle-check" class="bg-black border-gray-800" checked />
                                    <label for="spindle-check" class="text-sm text-gray-300">主軸轉速驗證</label>
                                  </div>
                                  <div class="flex items-center space-x-3">
                                    <input type="checkbox" id="feed-check" class="bg-black border-gray-800" checked />
                                    <label for="feed-check" class="text-sm text-gray-300">進給速度驗證</label>
                                  </div>
                                  <div class="flex items-center space-x-3">
                                    <input type="checkbox" id="tool-check" class="bg-black border-gray-800" checked />
                                    <label for="tool-check" class="text-sm text-gray-300">刀長補償檢查</label>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div class="flex justify-end space-x-3 mt-6 pt-4 border-t border-gray-800">
                          <button @click="closeConfig" class="px-6 py-2 text-gray-400 hover:text-gray-300 transition-colors">
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

                    <!-- Results Section - Full Chat Width -->
                    <div v-if="message.showResults" class="mt-4 border-t border-gray-800 pt-4">
                      <div class="bg-gray-900 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-4">
                          <h4 class="font-medium text-white text-lg">優化結果</h4>
                          <button @click="closeResults" class="text-gray-400 hover:text-gray-300">
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
                              
                              <div class="grid grid-cols-2 gap-6">
                                <div class="bg-gray-800 rounded-lg p-4">
                                  <h5 class="text-white font-medium mb-3">效能比較</h5>
                                  <div class="space-y-3">
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">原始加工時間：</span>
                                      <span class="text-white text-sm">45.2 分鐘</span>
                                    </div>
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">優化後加工時間：</span>
                                      <span class="text-white text-sm">34.8 分鐘</span>
                                    </div>
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">材料移除率：</span>
                                      <span class="text-white text-sm">12.5 cm³/min</span>
                                    </div>
                                    <div class="flex justify-between py-2">
                                      <span class="text-gray-400 text-sm">功耗：</span>
                                      <span class="text-white text-sm">平均 2.8 kW</span>
                                    </div>
                                  </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                  <h5 class="text-white font-medium mb-3">刀具分析</h5>
                                  <div class="space-y-3">
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">刀具磨損率：</span>
                                      <span class="text-white text-sm">0.02 mm/件</span>
                                    </div>
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">預期刀具壽命：</span>
                                      <span class="text-white text-sm">1,250 件</span>
                                    </div>
                                    <div class="flex justify-between py-2 border-b border-gray-700">
                                      <span class="text-gray-400 text-sm">切削力：</span>
                                      <span class="text-white text-sm">平均 245 N</span>
                                    </div>
                                    <div class="flex justify-between py-2">
                                      <span class="text-gray-400 text-sm">振動水準：</span>
                                      <span class="text-white text-sm">0.8 mm/s RMS</span>
                                    </div>
                                  </div>
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
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">效能指標時間軸</h5>
                                <div class="h-80 bg-gray-700 rounded flex items-center justify-center">
                                  <div class="text-center">
                                    <TrendingUp class="w-16 h-16 text-gray-400 mx-auto mb-4" />
                                    <p class="text-gray-400 mb-2">多指標效能圖表</p>
                                    <p class="text-gray-500 text-sm">進給速度、主軸轉速和品質指標隨時間變化</p>
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
                                      <div>G01 Z-2. F100</div>
                                      <div>G01 X10. F500</div>
                                      <div>G01 Y10.</div>
                                      <div>G01 X0.</div>
                                      <div>G01 Y0.</div>
                                      <div>G00 Z25.</div>
                                      <div>G01 Z-4. F100</div>
                                      <div>G01 X10. F500</div>
                                      <div>G01 Y10.</div>
                                      <div>G01 X0.</div>
                                      <div>G01 Y0.</div>
                                      <div>G00 Z25.</div>
                                      <div>M5</div>
                                      <div>M30</div>
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
                                      <div class="bg-gray-700">G01 Z-2. F150</div>
                                      <div class="bg-gray-700">G01 X10. F750</div>
                                      <div>G01 Y10.</div>
                                      <div>G01 X0.</div>
                                      <div>G01 Y0.</div>
                                      <div class="bg-gray-700">G01 Z-4. F150</div>
                                      <div class="bg-gray-700">G01 X10. F750</div>
                                      <div>G01 Y10.</div>
                                      <div>G01 X0.</div>
                                      <div>G01 Y0.</div>
                                      <div>G00 Z25.</div>
                                      <div>M5</div>
                                      <div>M30</div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">主要變更摘要</h5>
                                <div class="space-y-2">
                                  <div class="flex items-center space-x-3 p-2 bg-gray-700 rounded">
                                    <div class="w-2 h-2 bg-white rounded-full"></div>
                                    <span class="text-gray-300 text-sm">主軸轉速從 3000 提升至 3500 RPM</span>
                                  </div>
                                  <div class="flex items-center space-x-3 p-2 bg-gray-700 rounded">
                                    <div class="w-2 h-2 bg-white rounded-full"></div>
                                    <span class="text-gray-300 text-sm">下刀進給速度從 100 優化至 150 mm/min</span>
                                  </div>
                                  <div class="flex items-center space-x-3 p-2 bg-gray-700 rounded">
                                    <div class="w-2 h-2 bg-white rounded-full"></div>
                                    <span class="text-gray-300 text-sm">切削進給速度從 500 提升至 750 mm/min</span>
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
                              
                              <div class="grid grid-cols-2 gap-6">
                                <div class="bg-gray-800 rounded-lg p-4">
                                  <h5 class="text-white font-medium mb-3">模擬控制</h5>
                                  <div class="space-y-3">
                                    <div>
                                      <label class="block text-sm text-gray-400 mb-1">播放速度</label>
                                      <input type="range" class="w-full" min="0.1" max="5" step="0.1" value="1" />
                                    </div>
                                    <div class="flex items-center space-x-3">
                                      <input type="checkbox" id="show-toolpath" class="bg-black border-gray-800" checked />
                                      <label for="show-toolpath" class="text-sm text-gray-300">顯示刀路</label>
                                    </div>
                                    <div class="flex items-center space-x-3">
                                      <input type="checkbox" id="show-material" class="bg-black border-gray-800" checked />
                                      <label for="show-material" class="text-sm text-gray-300">顯示材料移除</label>
                                    </div>
                                  </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                  <h5 class="text-white font-medium mb-3">模擬統計</h5>
                                  <div class="space-y-2">
                                    <div class="flex justify-between">
                                      <span class="text-gray-400 text-sm">總操作數：</span>
                                      <span class="text-white text-sm">247</span>
                                    </div>
                                    <div class="flex justify-between">
                                      <span class="text-gray-400 text-sm">移除材料：</span>
                                      <span class="text-white text-sm">45.2 cm³</span>
                                    </div>
                                    <div class="flex justify-between">
                                      <span class="text-gray-400 text-sm">換刀次數：</span>
                                      <span class="text-white text-sm">3</span>
                                    </div>
                                    <div class="flex justify-between">
                                      <span class="text-gray-400 text-sm">快速移動：</span>
                                      <span class="text-white text-sm">28</span>
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
                                    <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                                      <AlertTriangle class="w-5 h-5 text-gray-400" />
                                      <span class="text-gray-300 text-sm">操作 15 主軸負載過高</span>
                                    </div>
                                    <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                                      <CheckCircle class="w-5 h-5 text-gray-400" />
                                      <span class="text-gray-300 text-sm">材料移除率：最佳</span>
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
                                    <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                                      <CheckCircle class="w-5 h-5 text-gray-400" />
                                      <span class="text-gray-300 text-sm">刀路優化已驗證</span>
                                    </div>
                                    <div class="flex items-center space-x-3 p-3 bg-gray-700 rounded">
                                      <CheckCircle class="w-5 h-5 text-gray-400" />
                                      <span class="text-gray-300 text-sm">G-code 語法驗證通過</span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                              
                              <div class="bg-gray-800 rounded-lg p-4">
                                <h5 class="text-white font-medium mb-3">詳細驗證報告</h5>
                                <div class="bg-black rounded p-4 text-sm font-mono text-gray-300 h-48 overflow-y-auto">
                                  <div>[INFO] 開始驗證程序...</div>
                                  <div>[PASS] 刀長補償檢查</div>
                                  <div>[PASS] 主軸轉速在機台限制內</div>
                                  <div>[PASS] 進給速度驗證</div>
                                  <div>[WARN] Z-4.0 處檢測到高切削力</div>
                                  <div>[PASS] 快速移動碰撞檢查</div>
                                  <div>[PASS] 工件座標系驗證</div>
                                  <div>[PASS] 換刀序列驗證</div>
                                  <div>[INFO] 驗證成功完成</div>
                                  <div>[INFO] 7 項檢查通過，1 項警告</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div class="flex justify-end space-x-3 mt-6 pt-4 border-t border-gray-800">
                          <button @click="closeResults" class="px-6 py-2 text-gray-400 hover:text-gray-300 transition-colors">
                            關閉
                          </button>
                          <button class="px-6 py-2 border border-gray-700 text-gray-300 rounded hover:border-gray-600 transition-colors">
                            匯出結果
                          </button>
                          <button 
                            @click="implementOptimization"
                            class="px-6 py-2 bg-white text-black rounded font-medium hover:bg-gray-100 transition-colors"
                          >
                            實施變更
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Chat Input -->
          <div class="bg-black border-t border-gray-800 p-4">
            <div class="flex space-x-3">
              <input 
                v-model="chatInput"
                @keypress.enter="sendMessage"
                type="text" 
                placeholder="提出問題或請求修改..."
                class="flex-1 px-3 py-2 bg-black border border-gray-800 rounded text-white placeholder-gray-500 text-sm focus:border-gray-600 focus:outline-none"
              />
              <button 
                @click="sendMessage"
                class="bg-white text-black px-4 py-2 rounded font-medium hover:bg-gray-100 transition-colors flex items-center space-x-2"
              >
                <Send class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Project Selection Modal -->
    <div v-if="showProjectModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
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
            @click="showProjectModal = false"
            class="px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors text-sm"
          >
            取消
          </button>
          <button 
            @click="createScenario"
            :disabled="!selectedProject || !selectedMachine || !scenarioName"
            class="px-4 py-2 bg-white text-black rounded text-sm font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            繼續
          </button>
        </div>
      </div>
    </div>

    <!-- Iteration Type Selection Modal -->
    <div v-if="showIterationModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
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
            @click="showIterationModal = false"
            class="px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors text-sm"
          >
            取消
          </button>
          <button 
            @click="startIteration"
            :disabled="!iterationType"
            class="px-4 py-2 bg-white text-black rounded text-sm font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            繼續
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { 
  Settings, LogOut, Plus, RotateCcw, Bot, Send, Play, Upload, ArrowRight,
  BarChart3, TrendingUp, Box, CheckCircle, AlertTriangle, Zap, Cpu,
  FileSpreadsheet, FileText, X
} from 'lucide-vue-next'

// Application State
const currentView = ref('landing')
const selectedPlatform = ref('')
const activeMode = ref('')
const showSimulationConfig = ref(false)
const selectedScenarioForIteration = ref(null)

// Authentication
const loginForm = reactive({
  username: '',
  password: ''
})

// Current scenario and messages
const currentScenario = ref(null)
const messages = ref([])

// Chat
const chatInput = ref('')

// Modals
const showProjectModal = ref(false)
const showIterationModal = ref(false)

// Form Data
const selectedProject = ref('')
const selectedMachine = ref('')
const scenarioName = ref('')
const iterationType = ref('')

// File Upload
const uploadedFiles = ref([])

// Tabs
const activeConfigTab = ref('machine')
const activeResultTab = ref('summary')

// Data
const projects = ref([
  { id: 'PRJ-001', name: '航太零件 A' },
  { id: 'PRJ-002', name: '汽車零件 B' },
  { id: 'PRJ-003', name: '醫療器材 C' }
])

const recentScenarios = ref([
  { id: 'SCN-001', name: '汽車零件加工專案', project: 'PRJ-001', date: '2024-01-15', type: '專案', status: 'completed', version: '1.2', completion: '92' },
  { id: 'SCN-002', name: '航空零件製造專案', project: 'PRJ-002', date: '2024-01-14', type: '專案', status: 'running', version: '2.1', completion: '88' },
  { id: 'SCN-003', name: '精密模具專案', project: 'PRJ-001', date: '2024-01-13', type: '專案', status: 'completed', version: '1.0', completion: '95' }
])

const simulations = ref([
  { id: 'SIM-001', name: '批次處理 A', project: 'PRJ-001', status: 'completed', created: '2024-01-15' },
  { id: 'SIM-002', name: '品質測試 B', project: 'PRJ-002', status: 'running', created: '2024-01-14' },
  { id: 'SIM-003', name: '效能測試 C', project: 'PRJ-003', status: 'failed', created: '2024-01-13' },
  { id: 'SIM-004', name: '負載測試 D', project: 'PRJ-001', status: 'pending', created: '2024-01-12' }
])

const configTabs = ref([
  { id: 'machine', name: '機台' },
  { id: 'material', name: '材料' },
  { id: 'tooling', name: '刀具' },
  { id: 'optimization', name: '優化' },
  { id: 'safety', name: '安全' }
])

const resultTabs = ref([
  { id: 'summary', name: '摘要' },
  { id: 'charts', name: '圖表' },
  { id: 'code', name: '程式碼差異' },
  { id: 'simulation', name: '模擬' },
  { id: 'validation', name: '驗證' }
])

// Computed
const getHeaderTitle = () => {
  if (activeMode.value === 'new') return '新專案生成'
  if (activeMode.value === 'iterate') return '迭代先前專案'
  if (activeMode.value === 'simulation') return '建立模擬'
  return 'CNC 優化助手'
}

const getHeaderSubtitle = () => {
  if (activeMode.value === 'new') return '建立新的優化專案'
  if (activeMode.value === 'iterate') return '修改和改進現有專案'
  if (activeMode.value === 'simulation') return '設定模擬參數'
  return '準備優化您的 CNC 程式'
}

const getStatusText = (status) => {
  const statusMap = {
    'completed': '已完成',
    'running': '執行中',
    'failed': '失敗',
    'pending': '待處理'
  }
  return statusMap[status] || status
}

// Methods
const navigateTo = (view, platform = '') => {
  currentView.value = view
  if (platform) selectedPlatform.value = platform
}

const login = () => {
  if (loginForm.username && loginForm.password) {
    currentView.value = 'main'
    // Don't initialize chat automatically
  }
}

const logout = () => {
  currentView.value = 'landing'
  loginForm.username = ''
  loginForm.password = ''
  messages.value = []
  currentScenario.value = null
  activeMode.value = ''
  selectedScenarioForIteration.value = null
}

const setActiveMode = (mode) => {
  activeMode.value = mode
  showSimulationConfig.value = false
  selectedScenarioForIteration.value = null
  
  if (mode === 'new') {
    initializeNewScenario()
  } else if (mode === 'iterate') {
    // Don't initialize chat, just show the scenario selection view
    messages.value = []
  } else if (mode === 'simulation') {
    initializeSimulation()
  }
}

const initializeNewScenario = () => {
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'assistant',
    content: '您好！我準備協助您建立新的優化專案。讓我們先選擇您的專案並配置參數。'
  })
  setTimeout(() => showProjectModal.value = true, 500)
}

const initializeSimulation = () => {
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'assistant',
    content: '歡迎使用模擬建立工作流程。您可以在表格中檢視現有模擬，或上傳 CNC 程式檔案來建立新模擬。'
  })
}

const selectScenarioForIteration = (scenario) => {
  selectedScenarioForIteration.value = scenario
  showIterationModal.value = true
}

const startIteration = () => {
  showIterationModal.value = false
  currentScenario.value = selectedScenarioForIteration.value
  
  messages.value = []
  messages.value.push({
    id: 1,
    type: 'user',
    content: `迭代專案 ${selectedScenarioForIteration.value.name}，進行 ${iterationType.value}`
  })
  
  setTimeout(() => {
    messages.value.push({
      id: 2,
      type: 'assistant',
      content: `載入專案「${selectedScenarioForIteration.value.name}」進行迭代。我將向您展示目前的配置，以便您針對${iterationType.value}進行調整。`,
      showConfig: true
    })
  }, 1000)
  
  // Reset form
  iterationType.value = ''
}

const createNewSimulation = () => {
  showSimulationConfig.value = true
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '建立新模擬'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '太好了！讓我們先上傳您的 CNC 程式檔案。我已準備了一些範本檔案供您下載，以確保格式正確。',
      showFileUpload: true
    })
  }, 1000)
}

const createScenario = () => {
  showProjectModal.value = false
  currentScenario.value = {
    name: scenarioName.value,
    project: selectedProject.value,
    machine: selectedMachine.value
  }
  
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: `建立新專案：${scenarioName.value}，專案 ${selectedProject.value}，機台 ${selectedMachine.value}`
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: `完美！我已為專案 ${selectedProject.value} 設定專案「${scenarioName.value}」。讓我向您展示基於您專案檔案的預設配置選項。`,
      showConfig: true
    })
  }, 1000)
  
  // Reset form
  selectedProject.value = ''
  selectedMachine.value = ''
  scenarioName.value = ''
}

const handleFileUpload = (event) => {
  const files = Array.from(event.target.files)
  uploadedFiles.value = [...uploadedFiles.value, ...files]
}

const removeFile = (fileToRemove) => {
  uploadedFiles.value = uploadedFiles.value.filter(file => file !== fileToRemove)
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const downloadTemplate = (type) => {
  // Simulate template download
  const filename = type === 'excel' ? 'cnc_template.xlsx' : 'cnc_template.csv'
  messages.value.push({
    id: messages.value.length + 1,
    type: 'assistant',
    content: `範本檔案「${filename}」已下載。請填入您的 CNC 程式資料並重新上傳。`
  })
}

const validateFiles = () => {
  if (uploadedFiles.value.length === 0) return
  
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: `驗證 ${uploadedFiles.value.length} 個上傳檔案`
  })
  
  // Simulate validation
  setTimeout(() => {
    const hasErrors = Math.random() > 0.7 // 30% chance of errors
    
    if (hasErrors) {
      messages.value.push({
        id: messages.value.length + 1,
        type: 'assistant',
        content: '資料驗證失敗。在上傳檔案中發現問題：<br/>• 檔案 1 缺少必要欄位<br/>• 檔案 2 G-code 語法無效<br/>請修正這些問題並重新上傳。',
        showFileUpload: true
      })
    } else {
      messages.value.push({
        id: messages.value.length + 1,
        type: 'assistant',
        content: '太好了！所有檔案都通過驗證。現在讓我們配置模擬參數。',
        showConfig: true
      })
    }
  }, 2000)
}

const runOptimization = () => {
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '使用目前配置執行優化'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '優化完成！這裡是您的結果，包含多個檢視的詳細分析。',
      showResults: true
    })
  }, 2000)
}

const implementOptimization = () => {
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: '實施優化變更'
  })
  
  setTimeout(() => {
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: '太棒了！優化的 CNC 程式已成功實施。變更已儲存到您的專案中。您現在可以執行此專案或基於這些結果建立變化版本。'
    })
    
    if (activeMode.value === 'simulation') {
      // Add to simulations table
      simulations.value.unshift({
        id: `SIM-${String(simulations.value.length + 1).padStart(3, '0')}`,
        name: currentScenario.value?.name || '新模擬',
        project: currentScenario.value?.project || 'PRJ-001',
        status: 'completed',
        created: new Date().toISOString().split('T')[0]
      })
      
      setTimeout(() => {
        showSimulationConfig.value = false
        messages.value.push({
          id: messages.value.length + 1,
          type: 'assistant',
          content: '模擬已新增到您的模擬執行表格中。您可以在主模擬儀表板中檢視。'
        })
      }, 1000)
    }
  }, 1000)
}

const sendMessage = () => {
  if (!chatInput.value.trim()) return
  
  messages.value.push({
    id: messages.value.length + 1,
    type: 'user',
    content: chatInput.value
  })
  
  const userMessage = chatInput.value.toLowerCase()
  chatInput.value = ''
  
  setTimeout(() => {
    let response = '我了解您的要求。'
    
    if (userMessage.includes('修改') || userMessage.includes('變更')) {
      response += '我可以協助您修改目前的配置。您想要調整哪些特定參數？'
    } else if (userMessage.includes('解釋') || userMessage.includes('為什麼')) {
      response += '讓我詳細解釋優化結果。改進來自於優化的進給速度、更好的刀路策略和減少非切削時間。'
    } else if (userMessage.includes('比較')) {
      response += '我可以向您展示原始和優化程式之間的詳細比較。您想要查看效能指標還是程式碼差異？'
    } else if (userMessage.includes('模擬')) {
      response += '對於模擬相關問題，我可以協助您了解 3D 視覺化、材料移除分析或碰撞檢測結果。'
    } else {
      response += '我還能如何協助您進行 CNC 優化？我可以協助參數調整、結果分析或建立新專案。'
    }
    
    messages.value.push({
      id: messages.value.length + 1,
      type: 'assistant',
      content: response
    })
  }, 1000)
}

const closeConfig = () => {
  // Find and update the message to hide config
  const configMessage = messages.value.find(m => m.showConfig)
  if (configMessage) {
    configMessage.showConfig = false
  }
}

const closeResults = () => {
  // Find and update the message to hide results
  const resultsMessage = messages.value.find(m => m.showResults)
  if (resultsMessage) {
    resultsMessage.showResults = false
  }
}
</script>

<style scoped>
/* Custom scrollbar for true black theme */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: #000000;
}

::-webkit-scrollbar-thumb {
  background: #404040;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #606060;
}
</style>