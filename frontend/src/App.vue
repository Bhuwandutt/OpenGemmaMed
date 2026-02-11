<script setup>
import { ref } from 'vue'

const prompt = ref('')
const response = ref('')
const loading = ref(false)

const askMedGemma = async () => {
  if (!prompt.value.trim()) return
  
  loading.value = true
  response.value = '' // Clear previous answer
  
  try {
    const res = await fetch('http://localhost:8000/ask/default_user', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt.value })
    })
    
    if (!res.ok) throw new Error('Backend error')
    
    const data = await res.json()
    response.value = data.answer
  } catch (err) {
    response.value = "⚠️ Connection Error: Ensure your FastAPI server is running on port 8000."
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="min-h-screen bg-slate-950 text-slate-200 font-sans p-4 md:p-12">
    <div class="max-w-3xl mx-auto">
      <header class="mb-10 text-center">
        <h1 class="text-4xl font-black text-emerald-500 tracking-tight">MED-GEMMA <span class="text-slate-500 font-light text-xl">v2.0</span></h1>
        <p class="text-slate-400 mt-2">Specialized Clinical LLM Interface • CachyOS Performance</p>
      </header>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl">
        <label class="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Clinical Query</label>
        <textarea 
          v-model="prompt"
          @keydown.enter.ctrl="askMedGemma"
          placeholder="Describe symptoms or paste a lab report summary..."
          class="w-full h-44 bg-slate-950 border border-slate-700 rounded-xl p-4 text-lg focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 outline-none transition-all resize-none"
        ></textarea>
        
        <button 
          @click="askMedGemma"
          :disabled="loading || !prompt"
          class="w-full mt-4 py-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-3"
        >
          <span v-if="loading" class="animate-pulse">⚡ Processing Inference...</span>
          <span v-else>Run Clinical Analysis</span>
        </button>
      </div>

      <transition name="fade">
        <div v-if="response" class="mt-8 bg-slate-900 border-l-4 border-emerald-500 p-8 rounded-r-2xl shadow-lg">
          <h2 class="text-emerald-500 font-bold mb-4 flex items-center gap-2">
            <span class="w-2 h-2 bg-emerald-500 rounded-full"></span> 
            AI CONSULTATION
          </h2>
          <div class="prose prose-invert max-w-none text-slate-300 leading-relaxed whitespace-pre-wrap">
            {{ response }}
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>

<style>
.fade-enter-active, .fade-leave-active { transition: opacity 0.5s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>