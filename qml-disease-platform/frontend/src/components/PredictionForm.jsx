import { useState } from 'react'

/**
 * Renders one numeric input per feature the trained model actually uses (fetched from
 * GET /diseases/{id}/metadata). This means the form NEVER hardcodes feature lists - if you
 * retrain a disease with a different feature_strategy in train.py, the form updates itself
 * automatically on next load, no frontend changes needed.
 */
export default function PredictionForm({ features, onSubmit, submitting }) {
  const [values, setValues] = useState(() =>
    Object.fromEntries(features.map((f) => [f, '']))
  )

  function handleChange(feature, raw) {
    setValues((prev) => ({ ...prev, [feature]: raw }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    const parsed = Object.fromEntries(
      Object.entries(values).map(([k, v]) => [k, parseFloat(v)])
    )
    onSubmit(parsed)
  }

  const allFilled = features.every((f) => values[f] !== '' && !Number.isNaN(parseFloat(values[f])))

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        {features.map((feature) => (
          <label key={feature} className="form-field">
            <span className="form-field-label">{formatFeatureName(feature)}</span>
            <input
              type="number"
              step="any"
              value={values[feature]}
              onChange={(e) => handleChange(feature, e.target.value)}
              placeholder="Enter value"
              required
            />
          </label>
        ))}
      </div>
      <button type="submit" className="submit-btn" disabled={!allFilled || submitting}>
        {submitting ? 'Running quantum circuit...' : 'Run prediction'}
      </button>
    </form>
  )
}

// Turns raw dataset column names (age, trestbps, chol...) into readable labels.
function formatFeatureName(name) {
  const overrides = {
    trestbps: 'Resting Blood Pressure',
    chol: 'Serum Cholesterol',
    thalach: 'Max Heart Rate Achieved',
    fbs: 'Fasting Blood Sugar > 120 mg/dl',
    exang: 'Exercise-Induced Angina',
    oldpeak: 'ST Depression (Exercise)',
    cp: 'Chest Pain Type',
    restecg: 'Resting ECG Result',
    ca: 'Major Vessels Colored',
    thal: 'Thalassemia',
  }
  if (overrides[name]) return overrides[name]
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
