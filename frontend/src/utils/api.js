import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export const optimizeParams = (data) =>
  api.post('/predict/optimize', data).then(r => r.data)

export const fetchModels = () =>
  api.get('/models/').then(r => r.data)

export const fetchProcesses = () =>
  api.get('/processes/').then(r => r.data)

export default api