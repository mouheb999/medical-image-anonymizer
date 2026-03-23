import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import API from '../api/axios'
import UploadZone from '../components/UploadZone'
import ResultViewer from '../components/ResultViewer'
import StatsPanel from '../components/StatsPanel'
import { getClassificationLabel } from '../utils/classificationLabel'

const Dashboard = () => {
  const { user } = useAuth()
  const { toast } = useToast()
  const [stats, setStats] = useState(null)
  const [recentLogs, setRecentLogs] = useState([])
  const [result, setResult] = useState(null)
  const [originalFile, setOriginalFile] = useState(null)

  useEffect(() => {
    loadStats()
    loadRecentLogs()
  }, [])

  const loadStats = async () => {
    try {
      const res = await API.get('/history?limit=100')
      const logs = res.data.data
      
      const totalImages = logs.length
      const successCount = logs.filter(l => l.status === 'success').length
      const failedCount = logs.filter(l => l.status === 'failed').length
      const avgTime = logs.reduce((sum, l) => sum + (l.processingTime || 0), 0) / (totalImages || 1)
      
      setStats({
        totalImages,
        successCount,
        failedCount,
        avgProcessingTime: Math.round(avgTime)
      })
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const loadRecentLogs = async () => {
    try {
      const res = await API.get('/history?limit=5')
      setRecentLogs(res.data.data)
    } catch (err) {
      console.error('Failed to load recent logs:', err)
    }
  }

  const handleResult = (data, file) => {
    setResult(data)
    setOriginalFile(file)
    loadStats()
    loadRecentLogs()
    toast.success('Image anonymized successfully!')
  }

  return (
    <div className="dashboard">
      <div className="container">
        <div className="welcome-section">
          <h1>Welcome back, <span>{user?.name}</span></h1>
          <p>MedSecure AI — Medical Image Anonymization Platform</p>
        </div>

        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{stats.totalImages}</div>
              <div className="stat-label">Total Images</div>
            </div>
            <div className="stat-card success">
              <div className="stat-value">{stats.successCount}</div>
              <div className="stat-label">Successful</div>
            </div>
            <div className="stat-card failed">
              <div className="stat-value">{stats.failedCount}</div>
              <div className="stat-label">Failed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{(stats.avgProcessingTime / 1000).toFixed(1)}s</div>
              <div className="stat-label">Avg Processing Time</div>
            </div>
          </div>
        )}

        <UploadZone onResult={handleResult} onFileSelect={setOriginalFile} />

        {result && originalFile && (
          <ResultViewer result={result} originalFile={originalFile} />
        )}

        {recentLogs.length > 0 && (
          <div className="recent-activity">
            <h2>Recent Activity</h2>
            <table className="log-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Filename</th>
                  <th>Classification</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recentLogs.map(log => (
                  <tr key={log._id}>
                    <td>{new Date(log.createdAt).toLocaleDateString()}</td>
                    <td>{log.originalFilename}</td>
                    <td style={{
                      maxWidth: '200px',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {getClassificationLabel(log.classification)}
                    </td>
                    <td>
                      <span className={`status-badge ${log.status}`}>
                        {log.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
