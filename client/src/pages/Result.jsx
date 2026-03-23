import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import API from '../api/axios'
import StatsPanel from '../components/StatsPanel'

const Result = () => {
  const { id } = useParams()
  const navigate = useNavigate()
  const [log, setLog] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadLog()
  }, [id])

  const loadLog = async () => {
    try {
      const res = await API.get(`/history/${id}`)
      setLog(res.data.data)
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load result')
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    if (log?.downloadUrl) {
      window.open(log.downloadUrl, '_blank')
    }
  }

  if (loading) return <div className="container"><div className="loading">Loading...</div></div>
  if (error) return <div className="container"><div className="error-message">{error}</div></div>
  if (!log) return <div className="container"><div>Log not found</div></div>

  return (
    <div className="result-page">
      <div className="container">
        <div className="page-header">
          <button onClick={() => navigate('/history')} className="btn-back">
            ← Back to History
          </button>
          <h1>Anonymization Result</h1>
        </div>

        <div className="result-info">
          <div className="info-row">
            <span className="label">Original File:</span>
            <span className="value">{log.originalFilename}</span>
          </div>
          <div className="info-row">
            <span className="label">Anonymized File:</span>
            <span className="value">{log.anonymizedFilename || 'N/A'}</span>
          </div>
          <div className="info-row">
            <span className="label">Date:</span>
            <span className="value">{new Date(log.createdAt).toLocaleString()}</span>
          </div>
          <div className="info-row">
            <span className="label">Status:</span>
            <span className={`status-badge ${log.status}`}>{log.status}</span>
          </div>
        </div>

        {log.status === 'success' && (
          <>
            <StatsPanel log={log} />
            
            {log.downloadUrl && (
              <div className="download-section">
                <button onClick={handleDownload} className="btn-primary">
                  Download Anonymized Image
                </button>
              </div>
            )}
          </>
        )}

        {log.status === 'failed' && log.errorMessage && (
          <div className="error-message">
            <strong>Error:</strong> {log.errorMessage}
          </div>
        )}
      </div>
    </div>
  )
}

export default Result
