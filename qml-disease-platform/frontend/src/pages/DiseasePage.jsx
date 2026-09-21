import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import PredictionForm from '../components/PredictionForm.jsx'
import ResultCard from '../components/ResultCard.jsx'
import BatchUpload from '../components/BatchUpload.jsx'
import BatchResultsTable from '../components/BatchResultsTable.jsx'
import { getDiseaseMetadata, predictDisease, predictBatch } from '../api/client.js'

const DISPLAY_NAMES = {
  heart: 'Heart Disease',
  cardiovascular: 'Cardiovascular Risk',
  diabetes: 'Diabetes',
  kidney: 'Kidney Disease',
  cancer: 'Cancer (Breast)',
}

export default function DiseasePage() {
  const { diseaseId } = useParams()
  const [metadata, setMetadata] = useState(null)
  const [metaError, setMetaError] = useState(null)
  const [result, setResult] = useState(null)
  const [predictError, setPredictError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [batchResult, setBatchResult] = useState(null)

  useEffect(() => {
    setMetadata(null)
    setMetaError(null)
    setResult(null)
    setPredictError(null)
    setBatchResult(null)

    getDiseaseMetadata(diseaseId)
      .then(setMetadata)
      .catch((err) => setMetaError(err.message))
  }, [diseaseId])

  async function handleSubmit(features) {
    setSubmitting(true)
    setPredictError(null)
    try {
      const res = await predictDisease(diseaseId, features)
      setResult(res)
    } catch (err) {
      setPredictError(err.message)
      setResult(null)
    } finally {
      setSubmitting(false)
    }
  }

  async function handleBatchSubmit(file) {
    const res = await predictBatch(diseaseId, file)
    setBatchResult(res)
  }

  return (
    <div className="disease-page">
      <h1 className="disease-title">{DISPLAY_NAMES[diseaseId] || diseaseId}</h1>
      {metadata && (
        <Link to={`/disease/${diseaseId}/performance`} className="performance-link">
          View model performance (accuracy, ROC, confusion matrix) →
        </Link>
      )}

      {metaError && (
        <div className="model-not-ready">
          <p>
            No trained model is available for this module yet. Train it first from the backend:
          </p>
          <pre>{`cd backend/src\npython train.py ${diseaseId}`}</pre>
          <p className="model-not-ready-detail">Server response: {metaError}</p>
        </div>
      )}

      {metadata && (
        <>
          <div className="disease-split">
            <div className="disease-form-col">
              <p className="feature-note">
                This model uses {metadata.n_qubits} qubits, one per selected feature below.
              </p>
              <PredictionForm
                features={metadata.required_features}
                onSubmit={handleSubmit}
                submitting={submitting}
              />
              <BatchUpload onSubmit={handleBatchSubmit} requiredFeatures={metadata.required_features} />
              <BatchResultsTable batchResult={batchResult} />
            </div>
            <div className="disease-result-col">
              <ResultCard result={result} error={predictError} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
