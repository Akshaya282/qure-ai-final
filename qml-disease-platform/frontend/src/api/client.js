// api/client.js
// Thin wrapper around fetch() for the FastAPI backend (backend/api/main.py).
// Change API_BASE if the backend is deployed somewhere other than local dev.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function handleResponse(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed with status ${res.status}`)
  }
  return res.json()
}

export async function listDiseases() {
  const res = await fetch(`${API_BASE}/diseases`)
  return handleResponse(res)
}

export async function getDiseaseMetadata(diseaseId) {
  const res = await fetch(`${API_BASE}/diseases/${diseaseId}/metadata`)
  return handleResponse(res)
}

export async function getDiseasePerformance(diseaseId) {
  const res = await fetch(`${API_BASE}/diseases/${diseaseId}/performance`)
  return handleResponse(res)
}

export async function predictBatch(diseaseId, file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/diseases/${diseaseId}/predict-batch`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(res)
}

// ---- General module (upload-your-own-dataset) ----

export async function generalInspect(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/general/inspect`, { method: 'POST', body: formData })
  return handleResponse(res)
}

export async function generalTrain(file, config) {
  const formData = new FormData()
  formData.append('file', file)
  Object.entries(config).forEach(([key, value]) => formData.append(key, value))
  const res = await fetch(`${API_BASE}/general/train`, { method: 'POST', body: formData })
  return handleResponse(res)
}

export async function generalMetadata() {
  const res = await fetch(`${API_BASE}/general/metadata`)
  return handleResponse(res)
}

export async function generalPerformance() {
  const res = await fetch(`${API_BASE}/general/performance`)
  return handleResponse(res)
}

export async function generalPredict(features) {
  const res = await fetch(`${API_BASE}/general/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })
  return handleResponse(res)
}

export async function generalPredictBatch(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/general/predict-batch`, { method: 'POST', body: formData })
  return handleResponse(res)
}

export async function predictDisease(diseaseId, features) {
  const res = await fetch(`${API_BASE}/diseases/${diseaseId}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })
  return handleResponse(res)
}
