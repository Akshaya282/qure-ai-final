import { Link } from 'react-router-dom'

const MODULES = [
  { id: 'heart', label: 'Heart Disease', desc: 'UCI Cleveland dataset · 13 clinical features · 6-qubit encoding' },
  { id: 'cardiovascular', label: 'Cardiovascular Risk', desc: 'Shares the heart-disease clinical dataset' },
  { id: 'diabetes', label: 'Diabetes', desc: 'Early-stage symptom questionnaire · 16 features · 6-qubit encoding' },
  { id: 'kidney', label: 'Kidney Disease', desc: 'Chronic kidney disease · 24 lab features · 6-qubit encoding' },
  { id: 'cancer', label: 'Cancer (Breast)', desc: 'Wisconsin diagnostic FNA features · 6-qubit encoding' },
  { id: 'brain_tumor', label: 'Brain Tumor', desc: 'MRI imaging · CNN bottleneck → 8-qubit encoding' },
]

export default function Home() {
  return (
    <div className="home">
      <section className="hero">
        <h1>
          A hybrid quantum-classical
          <br />
          platform for early disease detection.
        </h1>
        <p className="hero-sub">
          Classical pre-processing and feature selection feed a variational quantum circuit,
          benchmarked directly against classical baselines on the same held-out data.
          Built for SIH26139.
        </p>
      </section>

      <section className="module-grid">
        {MODULES.map((m) => (
          <Link key={m.id} to={`/disease/${m.id}`} className="module-row">
            <span className="module-label">{m.label}</span>
            <span className="module-desc">{m.desc}</span>
            <span className="module-arrow" aria-hidden="true">↗</span>
          </Link>
        ))}
        <Link to="/general" className="module-row module-row-general">
          <span className="module-label">Your Own Dataset</span>
          <span className="module-desc">Upload any patient CSV and train the full pipeline on it — no code required</span>
          <span className="module-arrow" aria-hidden="true">↗</span>
        </Link>
      </section>
    </div>
  )
}
