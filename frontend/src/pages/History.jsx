import { useState, useEffect } from 'react'
import { Trash2, Clock } from 'lucide-react'

export default function History() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    setHistory(JSON.parse(localStorage.getItem('miq_history') || '[]'))
  }, [])

  const clear = () => {
    localStorage.removeItem('miq_history')
    setHistory([])
  }

  return (
    <div className="page-wrapper">
      <div className="page-header-row">
        <div className="page-header" style={{ marginBottom: 0 }}>
          <h1>Run History</h1>
          <p>Past optimization and prediction runs</p>
        </div>
        {history.length > 0 && (
          <button onClick={clear} className="btn-ghost danger" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Trash2 size={13} /> Clear All
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 20px' }}>
          <Clock size={30} color="var(--muted)" style={{ margin: '0 auto 12px' }} />
          <div style={{ fontFamily: 'var(--font-body)', fontWeight: 600, color: 'var(--text-dim)', marginBottom: 4 }}>No history yet</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>Run predictions or optimizations to see results here</div>
        </div>
      ) : (
        <div className="history-list">
          {history.map((h, i) => {
            const date = new Date(h.ts)
            const r = h.result
            const energy = h.mode === 'optimize' ? r.optimized_predictions?.energy_consumption : r.energy_consumption
            const savings = h.mode === 'optimize' ? r.energy_savings_percent : null
            return (
              <div key={i} className="history-card">
                <div className="history-top">
                  <div className="history-meta">
                    <span className={`tag${h.mode === 'optimize' ? ' accent' : ''}`}>{h.mode}</span>
                    <span style={{ fontFamily: 'var(--font-body)', fontWeight: 600, fontSize: 13, color: 'var(--text)', textTransform: 'capitalize' }}>{h.process}</span>
                    <span style={{ color: 'var(--muted)' }}>·</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)', textTransform: 'capitalize' }}>{h.material?.replace('_', ' ')}</span>
                  </div>
                  <span className="history-time">
                    {date.toLocaleDateString()} {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
                <div className="history-stats">
                  <div>
                    <div className="history-stat-label">Engine</div>
                    <div className="history-stat-val">{h.modelId}</div>
                  </div>
                  <div>
                    <div className="history-stat-label">Energy</div>
                    <div className="history-stat-val accent">{energy?.toFixed(0) ?? '—'} W</div>
                  </div>
                  {savings !== null && (
                    <div>
                      <div className="history-stat-label">Savings</div>
                      <div className={`history-stat-val${savings > 0 ? ' accent' : ''}`}>{savings?.toFixed(1)}%</div>
                    </div>
                  )}
                  <div>
                    <div className="history-stat-label">Speed</div>
                    <div className="history-stat-val">{h.payload?.spindle_speed} RPM</div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}