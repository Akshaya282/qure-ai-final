import { useState } from 'react'
import MetricCards from '../components/MetricCards.jsx'
import ConfusionMatrix from '../components/ConfusionMatrix.jsx'
import RocChart from '../components/RocChart.jsx'
import PredictionForm from '../components/PredictionForm.jsx'
import ResultCard from '../components/ResultCard.jsx'
import BatchUpload from '../components/BatchUpload.jsx'
import BatchResultsTable from '../components/BatchResultsTable.jsx'
import {
  generalInspect, generalTrain, generalMetadata,
  generalPredict, generalPredictBatch,
} from '../api/client.js'

const FEATURE_STRATEGIES = [
  { value: 'mutual_info', label: 'Mutual Information (fast, recommended)' },
  { value: 'rfe', label: 'Recursive Feature Elimination (slower, accounts for interactions)' },
  { value: 'pso', label: 'Particle Swarm Optimization (swarm intelligence, slowest)' },
]

export default function GeneralPage() {
  const [step, setStep] = useState(1) // 1 = upload, 2 = configure, 3 = results/predict

  const [file, setFile] = useState(null)
  const [inspectData, setInspectData] = useState(null)
  const [inspectError, setInspectError] = useState(null)
  const [inspecting, setInspecting] = useState(false)

  const [targetColumn, setTargetColumn] = useState('')
  const [positiveLabel, setPositiveLabel] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [nQubits, setNQubits] = useState(6)
  const [featureStrategy, setFeatureStrategy] = useState('mutual_info')
  const [epochs, setEpochs] = useState(10)
  const [maxVqcSamples, setMaxVqcSamples] = useState(50)

  const [training, setTraining] = useState(false)
  const [trainError, setTrainError] = useState(null)
  const [performance, setPerformance] = useState(null)
  const [selectedModel, setSelectedModel] = useState(null)

  const [metadata, setMetadata] = useState(null)
  const [result, setResult] = useState(null)
  const [predictError, setPredictError] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [batchResult, setBatchResult] = useState(null)

  async function handleFileSelect(e) {
    const selected = e.target.files[0]
    if (!selected) return
    setFile(selected)
    setInspecting(true)
    setInspectError(null)
    setInspectData(null)
    try {
      const info = await generalInspect(selected)
      setInspectData(info)
      if (info.likely_target_columns.length > 0) {
        setTargetColumn(info.likely_target_columns[0])
      }
      setDisplayName(selected.name.replace(/\.csv$/i, ''))
      setStep(2)
    } catch (err) {
      setInspectError(err.message)
    } finally {
      setInspecting(false)
    }
  }

  const targetColumnInfo = inspectData?.columns.find((c) => c.name === targetColumn)

  async function handleTrain(e) {
    e.preventDefault()
    setTraining(true)
    setTrainError(null)
    try {
      const perf = await generalTrain(file, {
        target_column: targetColumn,
        positive_label: positiveLabel,
        display_name: displayName,
        n_qubits: nQubits,
        feature_strategy: featureStrategy,
        epochs,
        max_vqc_train_samples: maxVqcSamples,
      })
      setPerformance(perf)
      setSelectedModel(perf.best_model_by_accuracy)
      const meta = await generalMetadata()
      setMetadata(meta)
      setStep(3)
    } catch (err) {
      setTrainError(err.message)
    } finally {
      setTraining(false)
    }
  }

  async function handlePredict(features) {
    setPredicting(true)
    setPredictError(null)
    try {
      const res = await generalPredict(features)
      setResult(res)
    } catch (err) {
      setPredictError(err.message)
    } finally {
      setPredicting(false)
    }
  }

  async function handleBatchPredict(csvFile) {
    const res = await generalPredictBatch(csvFile)
    setBatchResult(res)
  }

  return (
    <div className="disease-page">
      <h1 className="disease-title">Train on Your Own Dataset</h1>
      <p className="feature-note">
        Upload any patient CSV — Qure AI runs the same pipeline used for every built-in disease
        module (feature selection, classical baselines, hybrid quantum VQC, QSVM) on your data,
        no code required.
      </p>

      {/* Step 1: Upload */}
      <div className="general-step">
        <h3 className="section-heading">1. Upload a CSV</h3>
        <input type="file" accept=".csv" onChange={handleFileSelect} disabled={inspecting} />
        {inspecting && <p className="feature-note">Reading file...</p>}
        {inspectError && <p className="batch-upload-error">{inspectError}</p>}
        {inspectData && (
          <p className="feature-note">
            {inspectData.n_rows} rows, {inspectData.n_columns} columns detected.
          </p>
        )}
      </div>

      {/* Step 2: Configure + train */}
      {inspectData && step >= 2 && (
        <div className="general-step">
          <h3 className="section-heading">2. Configure & train</h3>
          <form onSubmit={handleTrain} className="general-config-form">
            <label className="form-field">
              <span className="form-field-label">Target column (what you're predicting)</span>
              <select value={targetColumn} onChange={(e) => { setTargetColumn(e.target.value); setPositiveLabel('') }}>
                <option value="">— choose a column —</option>
                {inspectData.columns.map((c) => (
                  <option key={c.name} value={c.name}>{c.name} ({c.nunique} distinct values)</option>
                ))}
              </select>
            </label>

            {targetColumnInfo && (
              <label className="form-field">
                <span className="form-field-label">Which value means "disease present"?</span>
                {targetColumnInfo.unique_values ? (
                  <select value={positiveLabel} onChange={(e) => setPositiveLabel(e.target.value)}>
                    <option value="">— choose a value —</option>
                    {targetColumnInfo.unique_values.map((v) => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={positiveLabel}
                    onChange={(e) => setPositiveLabel(e.target.value)}
                    placeholder="Type the exact value, e.g. 1 or Yes"
                  />
                )}
              </label>
            )}

            <label className="form-field">
              <span className="form-field-label">Dataset / disease name</span>
              <input type="text" value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
            </label>

            <label className="form-field">
              <span className="form-field-label">Qubits (features to keep, 2–12)</span>
              <input type="number" min={2} max={12} value={nQubits} onChange={(e) => setNQubits(Number(e.target.value))} />
            </label>

            <label className="form-field">
              <span className="form-field-label">Feature selection strategy</span>
              <select value={featureStrategy} onChange={(e) => setFeatureStrategy(e.target.value)}>
                {FEATURE_STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </label>

            <label className="form-field">
              <span className="form-field-label">Quantum training epochs</span>
              <input type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
            </label>

            <label className="form-field">
              <span className="form-field-label">Max samples for quantum training</span>
              <input type="number" min={10} max={500} value={maxVqcSamples} onChange={(e) => setMaxVqcSamples(Number(e.target.value))} />
            </label>

            <button
              type="submit"
              className="submit-btn"
              disabled={!targetColumn || !positiveLabel || training}
            >
              {training ? 'Training (this can take 30–90s)...' : 'Train models'}
            </button>
          </form>
          {trainError && <p className="batch-upload-error">{trainError}</p>}
        </div>
      )}

      {/* Step 3: Results + predict */}
      {performance && step >= 3 && (
        <div className="general-step">
          <h3 className="section-heading">3. Results</h3>
          <p className="feature-note">
            Best model: <strong>{performance.best_model_by_accuracy}</strong>, evaluated on{' '}
            {performance.test_set_size} held-out rows.
          </p>

          <div className="model-tab-row">
            {Object.keys(performance.models).map((name) => (
              <button
                key={name}
                className={'model-tab' + (name === selectedModel ? ' model-tab-active' : '')}
                onClick={() => setSelectedModel(name)}
              >
                {name}
              </button>
            ))}
          </div>

          <MetricCards metrics={performance.models[selectedModel]} />

          <div className="performance-split">
            <div>
              <h3 className="section-heading">Confusion Matrix — {selectedModel}</h3>
              <ConfusionMatrix matrix={performance.models[selectedModel].confusion_matrix} />
            </div>
            <div>
              <h3 className="section-heading">ROC Curve — All Models</h3>
              <RocChart modelsPerf={performance.models} highlightModel={selectedModel} />
            </div>
          </div>

          <h3 className="section-heading">4. Predict</h3>
          {metadata && (
            <div className="disease-split">
              <div className="disease-form-col">
                <PredictionForm
                  features={metadata.required_features}
                  onSubmit={handlePredict}
                  submitting={predicting}
                />
                <BatchUpload onSubmit={handleBatchPredict} requiredFeatures={metadata.required_features} />
                <BatchResultsTable batchResult={batchResult} />
              </div>
              <div className="disease-result-col">
                <ResultCard result={result} error={predictError} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
