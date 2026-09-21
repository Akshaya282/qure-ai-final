const METRIC_DEFS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'sensitivity', label: 'Sensitivity (Recall)' },
  { key: 'specificity', label: 'Specificity' },
  { key: 'precision', label: 'Precision' },
  { key: 'f1_score', label: 'F1 Score' },
  { key: 'auc_roc', label: 'AUC-ROC' },
]

export default function MetricCards({ metrics }) {
  return (
    <div className="metric-cards">
      {METRIC_DEFS.map(({ key, label }) => (
        <div key={key} className="metric-card">
          <span className="metric-card-label">{label}</span>
          <span className="metric-card-value">
            {metrics[key] != null ? `${(metrics[key] * 100).toFixed(1)}%` : 'N/A'}
          </span>
        </div>
      ))}
    </div>
  )
}
