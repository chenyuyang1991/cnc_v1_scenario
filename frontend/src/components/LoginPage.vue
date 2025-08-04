<template>
  <div class="min-h-screen flex items-center justify-center p-4">
    <div class="bg-black border border-gray-800 rounded-lg p-8 w-full max-w-md">
      <div class="text-center mb-8">
        <div class="w-12 h-12 bg-white rounded-md flex items-center justify-center mx-auto mb-4">
          <Settings class="w-6 h-6 text-black" />
        </div>
        <h1 class="text-xl font-medium text-white">登入</h1>
        <p class="text-gray-400 mt-2 text-sm">存取 {{ selectedPlatform.toUpperCase() }} 平台</p>
      </div>
      
      <form @submit.prevent="handleLogin" class="space-y-4">
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
          :disabled="loading"
          class="w-full bg-white text-black py-2 rounded font-medium hover:bg-gray-100 disabled:bg-gray-300 transition-colors"
        >
          {{ loading ? '登入中...' : '登入' }}
        </button>
        
        <div v-if="error" class="text-red-400 text-center text-sm mt-2">
          {{ error }}
        </div>
      </form>
      
      <button 
        @click="$emit('navigate', 'landing')"
        class="w-full mt-4 text-gray-400 hover:text-gray-300 transition-colors text-sm"
      >
        ← 返回平台選擇
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { Settings } from 'lucide-vue-next'

const props = defineProps({
  selectedPlatform: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['login', 'navigate'])

const loginForm = reactive({
  username: '',
  password: ''
})

const loading = ref(false)
const error = ref('')

const handleLogin = async () => {
  if (!loginForm.username || !loginForm.password) {
    error.value = '請輸入用戶名和密碼'
    return
  }

  loading.value = true
  error.value = ''

  try {
    const { apiService } = await import('../services/apiService')
    
    const response = await apiService.login({
      username: loginForm.username,
      password: loginForm.password,
      platform: props.selectedPlatform
    })

    if (response.access_token) {
      localStorage.setItem('auth_token', response.access_token)
      localStorage.setItem('user_info', JSON.stringify(response.user))
      
      emit('login', {
        username: loginForm.username,
        password: loginForm.password,
        token: response.access_token,
        user: response.user,
        platform: props.selectedPlatform
      })
    } else {
      error.value = '登入失敗，請檢查用戶名和密碼'
    }
  } catch (err) {
    console.error('Login error:', err)
    error.value = err.message || '登入失敗，請重試'
  } finally {
    loading.value = false
  }
}
</script>  