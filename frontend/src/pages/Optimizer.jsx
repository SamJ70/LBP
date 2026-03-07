import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { predictParams, optimizeParams, fetchModels } from '../utils/api'
import ResultsPanel from '../components/ResultsPanel'
import { Cpu, Zap } from 'lucide-react'

const PROCESSES = [
  { id: 'turning',  name: 'CNC Turning',  icon: '🔄' },
  { id: 'milling',  name: 'CNC Milling',  icon: '⚙️' },
  { id: 'drilling', name: 'Drilling',     icon: '🔩' },
  { id: 'grinding', name: 'Grinding',     icon: '✨' },
]
const MATERIALS = [
  { value: 'aluminum',        label: 'Aluminum' },
  { value: 'steel_mild',      label: 'Mild Steel' },
  { value: 'steel_stainless', label: 'Stainless Steel' },
  { value: 'cast_iron',       label: 'Cast Iron' },
  { value: 'titanium',        label: 'Titanium' },
  { value: 'copper',          label: 'Copper' },
]
const TOOLS = [
  { value: 'hss',     label: 'HSS' },
  { value: 'carbide', label: 'Carbide' },
  { value: 'ceramic', label: 'Ceramic' },
  { value: 'cbn',     label: 'CBN' },
  { value: 'diamond', label: 'Diamond' },
]
const DEFAULTS = {
  turning:  { spindle_speed: 800,  feed_rate: 0.15, depth_of_cut: 1.5,  width_of_cut: '',  tool_diameter: 20 },
  milling:  { spindle_speed: 1200, feed_rate: 0.10, depth_of_cut: 2.0,  width_of_cut: 10,  tool_diameter: 16 },
  drilling: { spindle_speed: 600,  feed_rate: 0.20, depth_of_cut: 25,   width_of_cut: '',  tool_diameter: 10 },
  grinding: { spindle_speed: 1800, feed_rate: 0.01, depth_of_cut: 0.02, width_of_cut: 20,  tool_diameter: 150 },
}

function Field({ label, unit, children }) {
  return (
    <div className="field">
      <label className="field-label">{label} {unit && <span>({unit})</span>}</label>
      {children}
    </div>
  )
}

export default function Optimizer() {
  const [searchParams] = useSearchParams()
  const [process, setProcess] = useState(searchParams.get('process') || 'turning')
  const [material, setMaterial] = useState('steel_mild')
  const [toolMat, setToolMat] = useState('carbide')
  const [modelId, setModelId] = useState('physics_based')   // DEFAULT = physics, not AI
  const [coolant, setCoolant] = useState(false)
  const [models, setModels] = useState([])
  const [params, setParams] = useState(DEFAULTS['turning'])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('predict')

  useEffect(() => { fetchModels().then(setModels).catch(console.error) }, [])
  useEffect(() => { setParams(DEFAULTS[process]) }, [process])

  const setParam = (k, v) => setParams(p => ({ ...p, [k]: v }))

  const buildPayload = () => ({
    process_type:  process,
    material,
    tool_material: toolMat,
    spindle_speed: parseFloat(params.spindle_speed),
    feed_rate:     parseFloat(params.feed_rate),
    depth_of_cut:  parseFloat(params.depth_of_cut),
    width_of_cut:  params.width_of_cut ? parseFloat(params.width_of_cut) : null,
    tool_diameter: params.tool_diameter ? parseFloat(params.tool_diameter) : null,
    coolant_used:  coolant,
    model_id:      modelId,
  })

  const handleRun = async () => {
    setLoading(true); setError(null); setResults(null)
    try {
      const payload = buildPayload()
      const data = mode === 'optimize' ? await optimizeParams(payload) : await predictParams(payload)
      setResults({ mode, data })
      const hist = JSON.parse(localStorage.getItem('miq_history') || '[]')
      hist.unshift({ ts: Date.now(), mode, process, material, modelId, payload, result: data })
      localStorage.setItem('miq_history', JSON.stringify(hist.slice(0, 50)))
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  const showWidth = ['milling', 'grinding'].includes(process)

  return (
    <div className="page-wrapper-wide">
      <div className="page-header">
        <h1>Parameter Optimizer</h1>
        <p>Configure machining conditions and run physics-equation or AI-based analysis</p>
      </div>

      <div className="optimizer-layout">
        {/* ---- Left: Input Panel ---- */}
        <div className="form-panel">

          {/* Process */}
          <div className="card">
            <div className="card-title">Machining Process</div>
            <div className="process-chips">
              {PROCESSES.map(p => (
                <button
                  key={p.id}
                  onClick={() => setProcess(p.id)}
                  className={`process-chip${process === p.id ? ' active' : ''}`}
                >
                  <span>{p.icon}</span>
                  <span>{p.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Material & tool */}
          <div className="card">
            <div className="card-title">Material &amp; Tool</div>
            <Field label="Workpiece Material">
              <select className="input-field" value={material} onChange={e => setMaterial(e.target.value)}>
                {MATERIALS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </Field>
            <Field label="Tool Material">
              <select className="input-field" value={toolMat} onChange={e => setToolMat(e.target.value)}>
                {TOOLS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </Field>
            <div className="toggle-row" style={{ marginTop: 4 }}>
              <div className={`toggle${coolant ? ' on' : ''}`} onClick={() => setCoolant(c => !c)}>
                <div className="toggle-knob" />
              </div>
              <span className="toggle-label" onClick={() => setCoolant(c => !c)}>
                Coolant {coolant ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>

          {/* Parameters */}
          <div className="card">
            <div className="card-title">Cutting Parameters</div>
            <Field label="Spindle Speed" unit="RPM">
              <input type="number" className="input-field" value={params.spindle_speed}
                onChange={e => setParam('spindle_speed', e.target.value)} min={10} max={30000} step={50} />
            </Field>
            <Field label="Feed Rate" unit="mm/rev">
              <input type="number" className="input-field" value={params.feed_rate}
                onChange={e => setParam('feed_rate', e.target.value)} min={0.001} max={2} step={0.01} />
            </Field>
            <Field label="Depth of Cut" unit="mm">
              <input type="number" className="input-field" value={params.depth_of_cut}
                onChange={e => setParam('depth_of_cut', e.target.value)} min={0.001} max={20} step={0.05} />
            </Field>
            {showWidth && (
              <Field label="Width of Cut" unit="mm">
                <input type="number" className="input-field" value={params.width_of_cut}
                  onChange={e => setParam('width_of_cut', e.target.value)} min={0.1} max={100} step={0.5} />
              </Field>
            )}
            <Field label="Tool Diameter" unit="mm">
              <input type="number" className="input-field" value={params.tool_diameter}
                onChange={e => setParam('tool_diameter', e.target.value)} min={1} max={500} step={1} />
            </Field>
          </div>

          {/* Model selector */}
          <div className="card">
            <div className="card-title"><Cpu size={11} /> Prediction Engine</div>
            {models.length === 0 ? (
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>Loading models...</p>
            ) : (
              models.map(m => (
                <button
                  key={m.id}
                  onClick={() => setModelId(m.id)}
                  disabled={!m.available}
                  className={`model-btn${modelId === m.id ? ' active' : ''}`}
                >
                  <div className="model-btn-header">
                    <span className="model-btn-name">{m.name}</span>
                    <span className={`model-btn-badge${m.available ? ' ready' : ''}`}>
                      {m.available ? (m.type === 'physics' ? 'default' : 'ready') : 'unavail'}
                    </span>
                  </div>
                  <div className="model-btn-desc">{m.description}</div>
                </button>
              ))
            )}
          </div>

          {/* Mode + Run */}
          <div className="card">
            <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
              <button onClick={() => setMode('predict')} className={`btn-tab${mode === 'predict' ? ' active' : ''}`}>
                Predict Only
              </button>
              <button onClick={() => setMode('optimize')} className={`btn-tab${mode === 'optimize' ? ' active' : ''}`}>
                ⚡ Optimize
              </button>
            </div>
            <button onClick={handleRun} disabled={loading} className="btn-primary full">
              {loading
                ? <><div className="loading-dots"><span /><span /><span /></div> Running...</>
                : <><Zap size={14} /> {mode === 'optimize' ? 'Run Optimization' : 'Run Prediction'}</>
              }
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>

        {/* ---- Right: Results ---- */}
        <div>
          {results
            ? <ResultsPanel results={results} />
            : (
              <div className="empty-state">
                <div className="empty-icon">⚡</div>
                <div className="empty-title">No results yet</div>
                <div className="empty-sub">Configure parameters and run analysis</div>
              </div>
            )
          }
        </div>
      </div>
    </div>
  )
}