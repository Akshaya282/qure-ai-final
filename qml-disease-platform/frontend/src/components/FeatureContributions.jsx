export default function FeatureContributions({ explanation, note }) {
  if (!explanation || explanation.length === 0) return null

  const maxAbs = Math.max(...explanation.map((e) => Math.abs(e.contribution)), 0.0001)

  return (
    <div className="feature-contributions">
      <h4 className="fc-heading">Why this result — feature contributions</h4>
      <div className="fc-bars">
        {explanation.map((e) => {
          const widthPct = (Math.abs(e.contribution) / maxAbs) * 100
          const isPositive = e.contribution >= 0
          return (
            <div key={e.feature} className="fc-row">
              <span className="fc-label">{e.feature}</span>
              <div className="fc-track">
                <div
                  className={'fc-bar ' + (isPositive ? 'fc-bar-positive' : 'fc-bar-negative')}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <span className="fc-value">{e.value}</span>
            </div>
          )
        })}
      </div>
      <div className="fc-legend">
        <span><span className="fc-swatch fc-swatch-positive" /> pushes toward disease present</span>
        <span><span className="fc-swatch fc-swatch-negative" /> pushes toward disease unlikely</span>
      </div>
      {note && <p className="fc-note">{note}</p>}
    </div>
  )
}
