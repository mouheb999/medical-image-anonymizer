import { useState, useRef } from 'react'
import API from '../api/axios'
import '../styles/pathology.css'

const PathologyDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const acceptedTypes = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']

  const handleFileSelect = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!acceptedTypes.includes(ext)) {
      setError(`Unsupported file type. Accepted: ${acceptedTypes.join(', ')}`)
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setError('File size must be less than 50MB')
      return
    }

    setError('')
    setResult(null)
    setSelectedFile(file)

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0])
  }

  const handleDetect = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const res = await API.post('/pathology', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.message || 'Pathology detection failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // Colour scale mirrors backend severity buckets (High / Moderate / Low)
  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#ef4444'   // High
    if (confidence > 0.65) return '#f59e0b'  // Moderate
    return '#22c55e'                          // Low
  }

  return (
    <div className="pathology-page">
      <div className="container">
        <div className="pathology-header">
          <div className="header-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div>
            <h1>Pathology Detection</h1>
            <p>AI-powered chest X-ray analysis using TorchXRayVision DenseNet</p>
          </div>
        </div>

        <div className="pathology-content">
          {/* Upload Section */}
          <div className="pathology-upload-card">
            <h2>Upload Chest X-ray</h2>

            <div
              className={`pathology-dropzone ${isDragging ? 'dragging' : ''} ${preview ? 'has-file' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptedTypes.join(',')}
                onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
                style={{ display: 'none' }}
              />

              {preview ? (
                <div className="pathology-preview">
                  <img src={preview} alt="Selected X-ray" />
                  <p className="filename">{selectedFile?.name}</p>
                  <p className="filesize">{(selectedFile?.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div className="pathology-prompt">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <p>Drop a chest X-ray here or click to browse</p>
                  <span className="pathology-hint">Only chest X-rays are supported</span>
                </div>
              )}
            </div>

            {error && (
              <div className="pathology-error">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
                {error}
              </div>
            )}

            <div className="pathology-actions">
              <button
                onClick={handleDetect}
                disabled={!selectedFile || loading}
                className="btn-detect"
              >
                {loading ? (
                  <>
                    <span className="btn-spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                    </svg>
                    Analyze X-ray
                  </>
                )}
              </button>

              {selectedFile && (
                <button onClick={handleReset} className="btn-reset">
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* Results Section */}
          {result && (
            <div className="pathology-results-card">
              <h2>Detection Results</h2>

              <div className="pathology-warning">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                <span>{result.warning}</span>
              </div>

              {/* AI-Detected Patterns (safely hedged wording) */}
              <div className="pathologies-list">
                <h3>AI-Detected Patterns</h3>
                {result.pathologies && result.pathologies.length > 0 ? (
                  result.pathologies.map((p, idx) => (
                    <div key={idx} className="pathology-item">
                      <div className="pathology-item-header">
                        <span className="pathology-name">
                          {p.description || `Pattern possibly consistent with ${p.name.replace(/_/g, ' ')}`}
                        </span>
                        <span
                          className="pathology-confidence-badge"
                          style={{ backgroundColor: getConfidenceColor(p.confidence) }}
                        >
                          {p.severity || 'Low'}
                        </span>
                      </div>
                      <div className="pathology-bar-container">
                        <div
                          className="pathology-bar"
                          style={{
                            width: `${(p.confidence * 100).toFixed(0)}%`,
                            backgroundColor: getConfidenceColor(p.confidence)
                          }}
                        />
                      </div>
                      <span className="pathology-confidence-value">
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="pathology-none">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                    <p>No distinctive patterns detected above threshold</p>
                  </div>
                )}

                {/* Structured primary / secondary summary */}
                {result.summary && (result.summary.primary_finding || result.summary.secondary_finding) && (
                  <div className="pathology-summary">
                    {result.summary.primary_finding && (
                      <p><strong>Primary finding:</strong> {result.summary.primary_finding}</p>
                    )}
                    {result.summary.secondary_finding && (
                      <p><strong>Secondary finding:</strong> {result.summary.secondary_finding}</p>
                    )}
                  </div>
                )}
              </div>

              {/* Heatmap + Pseudo-Localization */}
              {result.heatmap && (
                <div className="heatmap-section">
                  <h3>Grad-CAM Heatmap &amp; Localization</h3>
                  <p className="heatmap-description">
                    Highlighted regions indicate areas the AI model focused on during analysis.
                    {result.bbox && ' A green bounding box marks the primary attention region.'}
                  </p>
                  <div className="heatmap-comparison">
                    <div className="heatmap-image-wrapper">
                      <span className="heatmap-label">Original</span>
                      {preview && <img src={preview} alt="Original X-ray" />}
                    </div>
                    <div className="heatmap-arrow">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="5" y1="12" x2="19" y2="12" />
                        <polyline points="12 5 19 12 12 19" />
                      </svg>
                    </div>
                    <div className="heatmap-image-wrapper">
                      <span className="heatmap-label">Model Attention Region</span>
                      <img
                        src={`data:image/png;base64,${result.heatmap}`}
                        alt="Grad-CAM Heatmap with localization"
                      />
                    </div>
                  </div>

                  {/* Bounding Box Info */}
                  {result.bbox && (
                    <div className="bbox-info">
                      <div className="bbox-badge">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="3" y="3" width="18" height="18" rx="2" />
                          <path d="M3 9h18M9 3v18" />
                        </svg>
                        <span>Region of Interest</span>
                      </div>
                      <div className="bbox-coords">
                        <span>x: {result.bbox[0]}</span>
                        <span>y: {result.bbox[1]}</span>
                        <span>w: {result.bbox[2]}px</span>
                        <span>h: {result.bbox[3]}px</span>
                      </div>
                    </div>
                  )}

                  {/* Localization disclaimer */}
                  {result.localization_note && (
                    <div className="localization-note">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="16" x2="12" y2="12" />
                        <line x1="12" y1="8" x2="12.01" y2="8" />
                      </svg>
                      <span>{result.localization_note}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Mandatory research-only disclaimer */}
              {result.disclaimer && (
                <div className="pathology-disclaimer">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                  <span>{result.disclaimer}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PathologyDetector
