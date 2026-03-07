import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Optimizer from './pages/Optimizer'
import History from './pages/History'
import Models from './pages/Models'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/optimize" element={<Optimizer />} />
        <Route path="/history" element={<History />} />
        <Route path="/models" element={<Models />} />
      </Routes>
    </Layout>
  )
}