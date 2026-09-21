import FeatureContributions from './FeatureContributions.jsx'

export default function ResultCard({ result, error }) {
  if (error) {
    return (
      <div className="result-card result-card-error">
        <h3>Prediction failed</h3>
        <p>{error}</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="result-card result-card-empty">
        <p>Fill in the patient values and run the prediction to see the hybrid quantum model's output here.</p>
      </div>
    )
  }

  const pct = Math.round(result.probability * 100)
  const isPositive = result.prediction === 'Disease Likely'

  return (
    <div className={'result-card ' + (isPositive ? 'result-positive' : 'result-negative')}>
      <div className="result-header">
        <span className="result-label">Model output</span>
        <span className="result-model-tag">{result.model_used}</span>
      </div>

      <div className="gauge">
        <div className="gauge-track">
          <div className="gauge-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="gauge-value">{pct}%</div>
      </div>

      <div className="result-verdict">{result.prediction}</div>
      <div className="result-confidence">Confidence band: {result.confidence_band}</div>

      <FeatureContributions explanation={result.explanation} note={result.explanation_note} />

      <p className="result-disclaimer">
        This is a research/hackathon prototype, not a certified diagnostic tool.
        Any clinical decision must involve a qualified physician.
      </p>
    </div>
  )
}
