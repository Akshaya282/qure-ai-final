export default function ConfusionMatrix({ matrix }) {
  const { TN, FP, FN, TP } = matrix

  return (
    <div className="confusion-matrix">
      <div className="cm-axis-label cm-axis-top">Predicted</div>
      <div className="cm-grid">
        <div className="cm-corner" />
        <div className="cm-col-label">No Disease</div>
        <div className="cm-col-label">Disease</div>

        <div className="cm-row-label">No Disease</div>
        <div className="cm-cell cm-cell-correct">
          <span className="cm-value">{TN}</span>
          <span className="cm-tag">True Negative</span>
        </div>
        <div className="cm-cell cm-cell-wrong">
          <span className="cm-value">{FP}</span>
          <span className="cm-tag">False Positive</span>
        </div>

        <div className="cm-row-label">Disease</div>
        <div className="cm-cell cm-cell-wrong">
          <span className="cm-value">{FN}</span>
          <span className="cm-tag">False Negative</span>
        </div>
        <div className="cm-cell cm-cell-correct">
          <span className="cm-value">{TP}</span>
          <span className="cm-tag">True Positive</span>
        </div>
      </div>
      <div className="cm-axis-label cm-axis-side">Actual</div>
    </div>
  )
}
