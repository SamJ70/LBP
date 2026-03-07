import { useNavigate } from 'react-router-dom'
import { Zap, TrendingDown, Shield, Cpu, ArrowRight, Activity } from 'lucide-react'

const PROCESSES = [
  { id: 'turning',  name: 'CNC Turning',  icon: '🔄', desc: 'Rotating workpiece, stationary tool' },
  { id: 'milling',  name: 'CNC Milling',  icon: '⚙️', desc: 'Multi-point rotating cutter' },
  { id: 'drilling', name: 'Drilling',     icon: '🔩', desc: 'Cylindrical hole creation' },
  { id: 'grinding', name: 'Grinding',     icon: '✨', desc: 'Abrasive precision finishing' },
]

const FEATURES = [
  { icon: Zap,          title: 'Energy Optimization',  desc: 'Minimize power using engineering equations + optional AI' },
  { icon: Shield,       title: 'Surface Quality',       desc: 'Predict Ra roughness via theoretical formulas' },
  { icon: TrendingDown, title: 'Tool Life Prediction',  desc: 'Taylor equation-based wear estimation' },
  { icon: Cpu,          title: 'Pluggable AI Models',   desc: 'Add ML/DL models anytime; equations run by default' },
]

export default function Dashboard() {
  const navigate = useNavigate()
  return (
    <div className="page-wrapper">
      <div style={{ marginBottom: 36 }}>
        <div className="online-badge">
          <div className="status-dot" />
          <span>System Online</span>
        </div>
        <h1 className="hero-title">
          ML-Based Machining<br />
          <span>Parameter Optimizer</span>
        </h1>
        <p className="hero-desc">
          Intelligent decision-support system for CNC parameter optimization —
          reducing energy consumption while maintaining surface quality and tool life.
          Physics equations run by default; plug in AI models when ready.
        </p>
        <div className="hero-actions">
          <button onClick={() => navigate('/optimize')} className="btn-primary">
            Start Optimizing <ArrowRight size={14} />
          </button>
          <button onClick={() => navigate('/models')} className="btn-ghost">
            View AI Models
          </button>
        </div>
      </div>

      <div style={{ marginBottom: 24 }}>
        <p className="section-header">Supported Processes</p>
        <div className="grid-2">
          {PROCESSES.map(p => (
            <button key={p.id} onClick={() => navigate(`/optimize?process=${p.id}`)} className="process-card">
              <div className="process-card-icon">{p.icon}</div>
              <div className="process-card-name">{p.name}</div>
              <div className="process-card-desc">{p.desc}</div>
              <div className="process-card-arrow">Optimize →</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="section-header">System Capabilities</p>
        <div className="grid-2">
          {FEATURES.map(f => (
            <div key={f.title} className="feature-card">
              <div className="feature-icon"><f.icon size={14} /></div>
              <div>
                <div className="feature-name">{f.title}</div>
                <div className="feature-desc">{f.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="info-bar">
        <div className="info-bar-left">
          <div className="status-dot" />
          Dept. of Mechanical &amp; Industrial Engineering · IIT Roorkee · MIC-300
        </div>
        <div className="info-bar-right">Supervisor: Prof. Rahul S. Mulik</div>
      </div>
    </div>
  )
}