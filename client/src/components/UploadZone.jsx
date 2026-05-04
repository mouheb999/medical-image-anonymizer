import { useState, useRef } from 'react'
import API from '../api/axios'
import AdvancedSettings from './AdvancedSettings'

const UploadZone = ({ onResult, onFileSelect }) => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)
  
  const [settings, setSettings] = useState({
    conf_threshold: 0.1,
    padding: 5,
    border_margin: 100,
    border_pct: 0.20
  })

  const acceptedTypes = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.bmp', '.tiff', '.tif']

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      validateAndSetFile(file)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      validateAndSetFile(file)
    }
  }

  const validateAndSetFile = (file) => {
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
    setSelectedFile(file)
    if (onFileSelect) onFileSelect(file)
  }

  const handleAnonymize = async () => {
    if (!selectedFile) return
    
    setLoading(true)
    setError('')
    
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('conf_threshold', settings.conf_threshold.toString())
      formData.append('padding', settings.padding.toString())
      formData.append('border_margin', settings.border_margin.toString())
      formData.append('border_pct', settings.border_pct.toString())
      
      const res = await API.post('/anonymize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      if (onResult) {
        onResult(res.data, selectedFile)
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Anonymization failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-section">
      <div
        className={`upload-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedTypes.join(',')}
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        
        {selectedFile ? (
          <div className="file-selected">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
              <polyline points="13 2 13 9 20 9"/>
            </svg>
            <p className="filename">{selectedFile.name}</p>
            <p className="filesize">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p>Drop your medical image here or click to browse</p>
            <p className="upload-hint">AI-powered anonymization in seconds</p>
            <div className="upload-formats">
              <span className="format-badge">DICOM</span>
              <span className="format-badge">JPEG</span>
              <span className="format-badge">PNG</span>
              <span className="format-badge">TIFF</span>
            </div>
          </div>
        )}
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <AdvancedSettings settings={settings} onChange={setSettings} />
      
      <button
        onClick={handleAnonymize}
        disabled={!selectedFile || loading}
        className="btn-primary btn-anonymize"
      >
        {loading ? 'Processing...' : 'Anonymize Image'}
      </button>
      
      {loading && (
        <div className="processing-info">
          <div className="spinner"></div>
          <p>Processing your image with AI pipeline...</p>
          <p className="processing-hint">This may take 10-30 seconds</p>
        </div>
      )}
    </div>
  )
}

export default UploadZone
