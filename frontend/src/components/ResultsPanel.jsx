import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'
import { TrendingDown, CheckCircle, AlertTriangle, Lightbulb } from 'lucide-react'

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

function PredictResults({ data }) {
  const radarData = [
    { metric: 'Efficiency', value: Math.min(100, (1 - data.energy_consumption / 2000) * 100) },
    { metric: 'Surface',    value: Math.min(100, (1 - data.surface_roughness / 10) * 100) },
    { metric: 'Tool Life',  value: Math.min(100, (1 - data.tool_wear_rate * 1000) * 100) },
    { metric: 'MRR',        value: Math.min(100, data.mrr / 100) },
    { metric: 'Confidence', value: data.confidence_score * 100 },
  ]
  const raColor = data.surface_roughness < 1.6 ? '' : data.surface_roughness < 3.2 ? 'warn' : 'danger'

  return (
    <div className="results-section">
      <div>
        <p className="section-header">Predicted Performance</p>
        <div className="metrics-grid">
          <MetricCard label="Energy Consumption" value={data.energy_consumption.toFixed(0)} unit="W" sub="Estimated machine power" />
          <MetricCard label="Surface Roughness" value={data.surface_roughness.toFixed(2)} unit="μm Ra" colorClass={raColor} />
          <MetricCard label="Tool Wear Rate" value={(data.tool_wear_rate * 1000).toFixed(3)} unit="mm/min ×10⁻³" />
          <MetricCard label="Material Removal Rate" value={data.mrr.toFixed(0)} unit="mm³/min" />
        </div>
      </div>

      <div className="card">
        <div className="card-title">Performance Radar</div>
        <ResponsiveContainer width="100%" height={200}>
          <RadarChart data={radarData} margin={{ top: 8, right: 28, left: 28, bottom: 8 }}>
            <PolarGrid stroke="#1e2229" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#8890a4', fontSize: 10, fontFamily: 'DM Mono' }} />
            <Radar dataKey="value" stroke="#00e5a0" fill="#00e5a0" fillOpacity={0.14} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <div className="card-title">Model Confidence</div>
        <ConfidenceBar score={data.confidence_score} />
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)', marginTop: 6 }}>
          Higher confidence = more reliable predictions. Physics model = deterministic; ML models = probabilistic.
        </p>
        {data.equations_used && (
          <div className="equation-note">{data.equations_used}</div>
        )}
      </div>
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
          {data.optimization_method && (
            <div className="equation-note">Method: {data.optimization_method}</div>
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
  const { mode, data } = results
  return (
    <div>
      <p className="results-header">
        {mode === 'optimize' ? '⚡ Optimization Results' : '📊 Prediction Results'}
      </p>
      {mode === 'optimize'
        ? <OptimizeResults data={data} />
        : <PredictResults data={data} />
      }
    </div>
  )
}