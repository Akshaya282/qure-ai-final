import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getDiseasePerformance } from '../api/client.js'
import MetricCards from '../components/MetricCards.jsx'
import ConfusionMatrix from '../components/ConfusionMatrix.jsx'
import RocChart from '../components/RocChart.jsx'

const DISPLAY_NAMES = {
  heart: 'Heart Disease',
  cardiovascular: 'Cardiovascular Risk',
  diabetes: 'Diabetes',
  kidney: 'Kidney Disease',
  cancer: 'Cancer (Breast)',
}

export default function PerformancePage() {
  const { diseaseId } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selectedModel, setSelectedModel] = useState(null)

  useEffect(() => {
    setData(null)
    setError(null)
    getDiseasePerformance(diseaseId)
      .then((res) => {
        setData(res)
        setSelectedModel(res.best_model_by_accuracy)
      })
      .catch((err) => setError(err.message))
  }, [diseaseId])

  if (error) {
    return (
      <div className="disease-page">
        <h1 className="disease-title">{DISPLAY_NAMES[diseaseId] || diseaseId} — Model Performance</h1>
        <div className="model-not-ready">
          <p>No performance data available for this module yet.</p>
          <pre>{`cd backend/src\npython train.py ${diseaseId}`}</pre>
          <p className="model-not-ready-detail">Server response: {error}</p>
        </div>
      </div>
    )
  }

  if (!data) return null

  const modelNames = Object.keys(data.models)
  const selected = data.models[selectedModel]

  return (
    <div className="disease-page">
      <div className="performance-header">
        <div>
          <h1 className="disease-title">{data.display_name} — Model Performance</h1>
          <p className="feature-note">
            Evaluated on a held-out test set of {data.test_set_size} patients
            ({(data.test_set_positive_rate * 100).toFixed(0)}% disease-positive).
            Best model by accuracy: <strong>{data.best_model_by_accuracy}</strong>.
          </p>
        </div>
        <Link to={`/disease/${diseaseId}`} className="back-link">← Back to prediction</Link>
      </div>

      <div className="model-tab-row">
        {modelNames.map((name) => (
          <button
            key={name}
            className={'model-tab' + (name === selectedModel ? ' model-tab-active' : '')}
            onClick={() => setSelectedModel(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <MetricCards metrics={selected} />

      <div className="performance-split">
        <div>
          <h3 className="section-heading">Confusion Matrix — {selectedModel}</h3>
          <ConfusionMatrix matrix={selected.confusion_matrix} />
          <p className="train-time-note">
            Training time: {selected.train_time_sec}s
            {selectedModel === 'HybridVQC' || selectedModel === 'QSVM'
              ? ' (quantum circuit simulation)'
              : ''}
          </p>
        </div>
        <div>
          <h3 className="section-heading">ROC Curve — All Models</h3>
          <RocChart modelsPerf={data.models} highlightModel={selectedModel} />
        </div>
      </div>

      <h3 className="section-heading">Full Comparison Table</h3>
      <div className="comparison-table-wrap">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Sensitivity</th>
              <th>Specificity</th>
              <th>Precision</th>
              <th>F1</th>
              <th>AUC-ROC</th>
              <th>Train Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {modelNames
              .slice()
              .sort((a, b) => data.models[b].accuracy - data.models[a].accuracy)
              .map((name) => (
                <tr key={name} className={name === selectedModel ? 'row-highlighted' : ''}>
                  <td>{name}</td>
                  <td>{(data.models[name].accuracy * 100).toFixed(1)}%</td>
                  <td>{(data.models[name].sensitivity * 100).toFixed(1)}%</td>
                  <td>{(data.models[name].specificity * 100).toFixed(1)}%</td>
                  <td>{(data.models[name].precision * 100).toFixed(1)}%</td>
                  <td>{(data.models[name].f1_score * 100).toFixed(1)}%</td>
                  <td>{data.models[name].auc_roc?.toFixed(3) ?? 'N/A'}</td>
                  <td>{data.models[name].train_time_sec}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
