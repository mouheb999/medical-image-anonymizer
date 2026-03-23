import { useState, useEffect } from 'react'
import { getClassificationLabel } from '../utils/classificationLabel'

const ResultViewer = ({ result, originalFile }) => {
  const [originalPreview, setOriginalPreview] = useState(null)

  useEffect(() => {
    if (originalFile) {
      const reader = new FileReader()
      reader.onload = (e) => setOriginalPreview(e.target.result)
      reader.readAsDataURL(originalFile)
    }
  }, [originalFile])

  const anonymizedUrl = result.preview_filename 
    ? `http://localhost:8000/result/${result.preview_filename}`
    : `http://localhost:8000/result/${result.output_filename}`

  const handleDownload = () => {
    window.open(`http://localhost:8000/result/${result.output_filename}`, '_blank')
  }

  return (
    <div className="result-viewer">
      <h2>Anonymization Complete</h2>
      
      <div className="image-comparison">
        <div className="image-container">
          <h3>Original</h3>
          {originalPreview && (
            <img src={originalPreview} alt="Original" />
          )}
          <p className="image-label">{originalFile?.name}</p>
        </div>
        
        <div className="comparison-arrow">→</div>
        
        <div className="image-container">
          <h3>Anonymized</h3>
          <img src={anonymizedUrl} alt="Anonymized" />
          <p className="image-label">{result.output_filename}</p>
        </div>
      </div>
      
      <div className="result-stats">
        <div className="stat-item">
          <span className="stat-label">Classification:</span>
          <span className="stat-value" style={{
            fontSize: '14px',
            fontWeight: '600',
            color: '#1a2332',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            maxWidth: '100%',
            display: 'block'
          }}>
            {getClassificationLabel(result.classification)}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Confidence:</span>
          <span className="stat-value">{(result.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Format:</span>
          <span className="stat-value">{result.format}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Regions Redacted:</span>
          <span className="stat-value">{result.redacted}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Processing Time:</span>
          <span className="stat-value">{(result.processingTime / 1000).toFixed(2)}s</span>
        </div>
      </div>
      
      <button onClick={handleDownload} className="btn-primary">
        Download Anonymized Image
      </button>
    </div>
  )
}

export default ResultViewer
