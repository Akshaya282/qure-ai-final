import { NavLink } from 'react-router-dom'

// The disease list mirrors backend/src/config.py's DISEASE_REGISTRY.
// Keeping this in sync manually is fine at this scale (6 entries); if the registry grows,
// fetch it from GET /diseases instead (see api/client.js: listDiseases()).
const DISEASES = [
  { id: 'heart', label: 'Heart' },
  { id: 'cardiovascular', label: 'Cardiovascular' },
  { id: 'diabetes', label: 'Diabetes' },
  { id: 'kidney', label: 'Kidney' },
  { id: 'cancer', label: 'Cancer' },
  { id: 'brain_tumor', label: 'Brain Tumor' },
]

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="brand">
          <svg className="brand-mark" viewBox="0 0 40 24" aria-hidden="true">
            {/* A minimal 3-qubit circuit line motif, standing in for a logo mark */}
            <line x1="2" y1="4" x2="38" y2="4" stroke="currentColor" strokeWidth="1.4" />
            <line x1="2" y1="12" x2="38" y2="12" stroke="currentColor" strokeWidth="1.4" />
            <line x1="2" y1="20" x2="38" y2="20" stroke="currentColor" strokeWidth="1.4" />
            <circle cx="13" cy="4" r="3" fill="currentColor" />
            <circle cx="24" cy="12" r="3" fill="currentColor" />
            <circle cx="17" cy="20" r="3" fill="currentColor" />
          </svg>
          <span className="brand-text">
            <span className="brand-title">Qure AI</span>
            <span className="brand-sub">Hybrid Quantum Diagnostics</span>
          </span>
        </NavLink>

        <nav className="tab-row" aria-label="Disease modules">
          {DISEASES.map((d) => (
            <NavLink
              key={d.id}
              to={`/disease/${d.id}`}
              className={({ isActive }) => 'tab-link' + (isActive ? ' tab-link-active' : '')}
            >
              {d.label}
            </NavLink>
          ))}
          <NavLink
            to="/general"
            className={({ isActive }) => 'tab-link tab-link-general' + (isActive ? ' tab-link-active' : '')}
          >
            + Your Dataset
          </NavLink>
        </nav>
      </div>
    </header>
  )
}
