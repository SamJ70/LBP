import { useState, useEffect } from 'react'
import { fetchModels } from '../utils/api'
import { CheckCircle, XCircle } from 'lucide-react'

const TYPE_EMOJI = {
  physics: '🔬',
  sklearn: '🤖',
  llm:     '💬',
  custom_nn: '🧠',
}

export default function Models() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error).finally(() => setLoading(false))
  }, [])

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <h1>AI Models</h1>
        <p>All registered prediction engines. Physics equations run by default. Add AI models anytime.</p>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <div className="loading-dots" style={{ '--bg': 'var(--accent)' }}>
            <span style={{ background: 'var(--accent)' }} />
            <span style={{ background: 'var(--accent)' }} />
            <span style={{ background: 'var(--accent)' }} />
          </div>
        </div>
      ) : (
        <div className="models-list">
          {models.map(m => (
            <div key={m.id} className={`model-card${m.available ? '' : ' unavailable'}`}>
              <div className={`model-type-icon ${m.type}`}>
                {TYPE_EMOJI[m.type] || '🤖'}
              </div>
              <div className="model-info">
                <div className="model-header">
                  <span className="model-name">{m.name}</span>
                  <span className="tag">{m.type}</span>
                  <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5 }}>
                    {m.available
                      ? <><CheckCircle size={13} color="var(--accent)" /><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent)' }}>Available</span></>
                      : <><XCircle size={13} color="var(--muted)" /><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>Unavailable</span></>
                    }
                  </div>
                </div>
                <p className="model-desc">{m.description}</p>
                <div className="model-tags">
                  {m.supported_processes?.map(p => (
                    <span key={p} className="tag" style={{ textTransform: 'capitalize' }}>{p}</span>
                  ))}
                  {m.accuracy_metrics && Object.entries(m.accuracy_metrics).map(([k, v]) => (
                    <span key={k} className="tag accent">{k}: {v}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}

          <div className="model-add-card">
            <div className="add-icon">+</div>
            <div>
              <div style={{ fontFamily: 'var(--font-body)', fontWeight: 600, fontSize: 13, color: 'var(--text-dim)', marginBottom: 3 }}>
                Add Your Trained Model
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>
                Copy <code>custom_model_template.py</code>, implement <code>predict()</code>, register in <code>model_registry.py</code>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="how-to-card">
        <div className="card-title" style={{ marginBottom: 14 }}>How to Plug In Your Trained Model</div>
        <div className="how-to-steps">
          {[
            'Train your model (sklearn, PyTorch, XGBoost, ONNX — any framework)',
            'Copy backend/app/ml_models/custom_model_template.py → my_model.py',
            'Implement predict() loading your saved model file',
            'In model_registry.py add: "my_model": MyModelClass',
            'Restart backend — it appears in the engine selector instantly',
          ].map((t, i) => (
            <div key={i} className="how-to-step">
              <div className="step-num">{i + 1}</div>
              <div className="step-text">{t}</div>
            </div>
          ))}
        </div>
      </div>

      {/* HuggingFace integration guide */}
      <div className="how-to-card" style={{ marginTop: 14, borderColor: 'rgba(255,149,0,0.2)' }}>
        <div className="card-title" style={{ marginBottom: 14 }}>HuggingFace API Integration (Runs on HF servers, NOT your laptop)</div>
        <div className="how-to-steps">
          {[
            'Go to huggingface.co/settings/tokens → create a free token',
            'Copy backend/.env.example → .env and set HUGGINGFACE_API_KEY=hf_xxx',
            'The huggingface_llm engine calls the API — model runs on HF cloud servers',
            'No GPU, no download, no RAM usage on your machine at all',
            'For your own fine-tuned HF model: change HF_LLM_MODEL in huggingface_model.py',
          ].map((t, i) => (
            <div key={i} className="how-to-step">
              <div className="step-num" style={{ background: 'rgba(255,149,0,0.15)', color: 'var(--warn)' }}>{i + 1}</div>
              <div className="step-text">{t}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}