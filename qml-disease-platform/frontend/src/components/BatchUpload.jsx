import { useState } from 'react'

export default function BatchUpload({ onSubmit, requiredFeatures }) {
  const [file, setFile] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) return
    setSubmitting(true)
    setError(null)
    try {
      await onSubmit(file)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="batch-upload">
      <h4 className="fc-heading">Or upload a CSV of multiple patients</h4>
      {requiredFeatures && (
        <p className="batch-upload-hint">
          CSV must include columns: <code>{requiredFeatures.join(', ')}</code> (extra columns, like a patient ID, are ignored)
        </p>
      )}
      <form onSubmit={handleSubmit} className="batch-upload-form">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0] || null)}
        />
        <button type="submit" className="submit-btn submit-btn-small" disabled={!file || submitting}>
          {submitting ? 'Processing...' : 'Predict for all rows'}
        </button>
      </form>
      {error && <p className="batch-upload-error">{error}</p>}
    </div>
  )
}
