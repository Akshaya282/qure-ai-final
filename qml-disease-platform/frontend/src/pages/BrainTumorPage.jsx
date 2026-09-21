import { useState } from 'react'
import ResultCard from '../components/ResultCard.jsx'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function BrainTumorPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  function handleFileChange(e) {
    const selected = e.target.files[0]
    if (!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResult(null)
    setError(null)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) return
    setSubmitting(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/diseases/brain_tumor/predict-image`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed with status ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="disease-page">
      <h1 className="disease-title">Brain Tumor</h1>
      <p className="feature-note">
        This module runs an MRI slice through a ResNet-18 encoder, compresses it to 8 values, then
        feeds those into the same 8-qubit variational quantum classifier used across the platform.
      </p>

      <div className="disease-split">
        <div className="disease-form-col">
          <form className="upload-form" onSubmit={handleSubmit}>
            <label className="upload-dropzone">
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
              {preview ? (
                <img src={preview} alt="Selected MRI scan" className="upload-preview" />
              ) : (
                <span>Click to select an MRI image (JPG/PNG)</span>
              )}
            </label>
            <button type="submit" className="submit-btn" disabled={!file || submitting}>
              {submitting ? 'Running CNN + quantum circuit...' : 'Run prediction'}
            </button>
          </form>
        </div>
        <div className="disease-result-col">
          <ResultCard result={result} error={error} />
        </div>
      </div>
    </div>
  )
}
