import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'
import { CheckCircle, AlertTriangle, Lightbulb } from 'lucide-react'

function MetricCard({ label, value, unit, sub, colorClass = '', change }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value-row">
        <span className={`metric-value ${colorClass}`}>{value}</span>
        <span className="metric-unit">{unit}</span>
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
      {change !== undefined && (
        <div className={`metric-change ${change < 0 ? 'pos' : 'neg'}`}>
          {change > 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}% vs original
        </div>
      )}
    </div>
  )
}

function ConfidenceBar({ score }) {
  const pct = score * 100
  const cls = pct >= 80 ? '' : pct >= 60 ? 'medium' : 'low'
  const dotColor = pct >= 80 ? 'var(--accent)' : pct >= 60 ? 'var(--warn)' : 'var(--muted)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 7, height: 7, borderRadius: '50%', background: dotColor, flexShrink: 0 }} />
      <div className="confidence-bar-track">
        <div className={`confidence-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)', flexShrink: 0 }}>
        {pct.toFixed(0)}%
      </span>
    </div>
  )
}


function OptimizeResults({ data }) {
  const orig = data.original_predictions
  const opt  = data.optimized_predictions

  const barData = [
    { name: 'Energy (W)',  original: Math.round(orig.energy_consumption),  optimized: Math.round(opt.energy_consumption) },
    { name: 'Ra ×10',      original: +(orig.surface_roughness * 10).toFixed(1), optimized: +(opt.surface_roughness * 10).toFixed(1) },
    { name: 'MRR /100',    original: +(orig.mrr / 100).toFixed(1),          optimized: +(opt.mrr / 100).toFixed(1) },
  ]

  return (
    <div className="results-section">
      {/* Savings banner */}
      <div className={`savings-banner${data.energy_savings_percent > 0 ? ' positive' : ''}`}>
        <div className="savings-icon">📉</div>
        <div>
          <div className="savings-pct">{data.energy_savings_percent.toFixed(1)}%</div>
          <div className="savings-label">Energy savings achieved</div>
        </div>
        <div className={`savings-quality${data.quality_maintained ? ' ok' : ' warn'}`}>
          {data.quality_maintained
            ? <><CheckCircle size={14} /> Quality maintained</>
            : <><AlertTriangle size={14} /> Quality trade-off</>
          }
        </div>
      </div>

      {/* Before/after metrics */}
      <div>
        <p className="section-header">Before vs After</p>
        <div className="metrics-grid">
          <MetricCard label="Energy: Original"  value={orig.energy_consumption.toFixed(0)} unit="W" colorClass="warn" />
          <MetricCard label="Energy: Optimized" value={opt.energy_consumption.toFixed(0)}  unit="W"
            change={((opt.energy_consumption - orig.energy_consumption) / orig.energy_consumption) * 100} />
          <MetricCard label="Ra: Original"  value={orig.surface_roughness.toFixed(2)} unit="μm" colorClass="dim" />
          <MetricCard label="Ra: Optimized" value={opt.surface_roughness.toFixed(2)}  unit="μm"
            change={((opt.surface_roughness - orig.surface_roughness) / orig.surface_roughness) * 100} />
        </div>
      </div>

      {/* Bar chart */}
      <div className="card">
        <div className="card-title">Comparison Chart</div>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
            <XAxis dataKey="name" tick={{ fill: '#8890a4', fontSize: 9, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#8890a4', fontSize: 9 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#111318', border: '1px solid #1e2229', borderRadius: 8, fontFamily: 'DM Mono', fontSize: 11 }} />
            <Bar dataKey="original" name="Original" fill="#ff9500" radius={[3,3,0,0]} />
            <Bar dataKey="optimized" name="Optimized" fill="#00e5a0" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Parameter changes */}
      {data.optimization_notes?.length > 0 && (
        <div className="card">
          <div className="card-title">Parameter Changes</div>
          <div className="notes-list">
            {data.optimization_notes.map((n, i) => (
              <div key={i} className="note-row">
                <span className="note-arrow">→</span>
                <span className="note-text">{n}</span>
              </div>
            ))}
          </div>

          {data.applied_constraints && (
            <div style={{ marginTop: 16 }}>
              <div className="card-title" style={{ fontSize: 11, marginBottom: 8, color: 'var(--text-dim)' }}>Applied Constraints</div>
              <div className="notes-list">
                <div className="note-row">
                  <span className="note-arrow">→</span>
                  <span className="note-text">Max Ra (≤ {data.applied_constraints.max_surface_roughness_factor?.toFixed(2)}x): <span style={{color: 'var(--muted)'}}>{data.original_params?.max_surface_roughness_factor ? '(User Defined)' : '(Default)'}</span></span>
                </div>
                <div className="note-row">
                  <span className="note-arrow">→</span>
                  <span className="note-text">Min MRR (≥ {data.applied_constraints.min_mrr_factor?.toFixed(2)}x): <span style={{color: 'var(--muted)'}}>{data.original_params?.min_mrr_factor ? '(User Defined)' : '(Default)'}</span></span>
                </div>
                <div className="note-row">
                  <span className="note-arrow">→</span>
                  <span className="note-text">Min Tool Life (≥ {typeof data.applied_constraints.min_tool_life === 'number' ? data.applied_constraints.min_tool_life.toFixed(1) + ' min' : data.applied_constraints.min_tool_life}): <span style={{color: 'var(--muted)'}}>{data.original_params?.min_tool_life ? '(User Defined)' : '(Default)'}</span></span>
                </div>
              </div>
            </div>
          )}

          {data.optimization_method && (
            <div className="equation-note" style={{ marginTop: 12 }}>Method: {data.optimization_method}</div>
          )}
        </div>
      )}

      {/* AI / expert advice */}
      {data.ai_advice && (
        <div className="advice-card">
          <div className="advice-header">
            <Lightbulb size={13} color="var(--accent)" />
            <span className="advice-title">Expert Recommendations</span>
          </div>
          <p className="advice-text">{data.ai_advice}</p>
          <p className="advice-footer">Engine: {data.model_used}</p>
        </div>
      )}
    </div>
  )
}

export default function ResultsPanel({ results }) {
  const { data } = results
  return (
    <div>
      <p className="results-header">⚡ Optimization Results</p>
      <OptimizeResults data={data} />
    </div>
  )
}