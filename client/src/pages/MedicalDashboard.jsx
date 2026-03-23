import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import API from '../api/axios'
import UploadZone from '../components/UploadZone'
import ResultViewer from '../components/ResultViewer'
import StatsPanel from '../components/StatsPanel'
import { getClassificationLabel } from '../utils/classificationLabel'

const MedicalDashboard = () => {
  const { user } = useAuth()
  const { toast } = useToast()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [recentLogs, setRecentLogs] = useState([])
  const [allImages, setAllImages] = useState([])
  const [imagesLoading, setImagesLoading] = useState(false)
  const [previewModal, setPreviewModal] = useState(false)
  const [previewImage, setPreviewImage] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  useEffect(() => {
    loadStats()
    loadRecentLogs()
  }, [])

  useEffect(() => {
    if (activeTab === 'all-images') {
      loadAllImages()
    }
  }, [activeTab])

  const loadStats = async () => {
    try {
      const res = await API.get('/history?limit=1000')
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

  const loadAllImages = async () => {
    setImagesLoading(true)
    try {
      const res = await API.get('/history/images/all')
      setAllImages(res.data.data)
    } catch (err) {
      console.error('Failed to load all images:', err)
    } finally {
      setImagesLoading(false)
    }
  }

  const handleImageDownload = async (imageId, filename) => {
    try {
      console.log('[DEBUG] Download requested for imageId:', imageId)
      console.log('[DEBUG] Token in localStorage:', localStorage.getItem('token') ? 'EXISTS' : 'MISSING')
      
      const res = await API.get(`/history/images/${imageId}/download`)
      console.log('[DEBUG] Download response:', res.data)
      
      if (res.data.presigned_url) {
        const link = document.createElement('a')
        link.href = res.data.presigned_url
        link.download = filename
        link.target = '_blank'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      } else {
        console.error('[DEBUG] No presigned_url in response:', res.data)
        toast.error('Download failed: no URL returned')
      }
    } catch (err) {
      console.error('[DEBUG] Download error full details:', {
        message: err.message,
        status: err?.response?.status,
        data: err?.response?.data,
        config: err?.config
      })
      toast.error(`Download failed: ${err?.response?.data?.message || err.message}`)
    }
  }

  const handleViewImage = async (imageId, filename, classification, redacted, uploadDate, uploadedBy) => {
    setPreviewLoading(true)
    setPreviewModal(true)
    try {
      console.log('[DEBUG] View requested for imageId:', imageId)
      const res = await API.get(`/history/images/${imageId}/download`)
      console.log('[DEBUG] View response:', res.data)
      setPreviewImage({
        url: res.data.presigned_url,
        filename: filename,
        imageId: imageId,
        classification: classification,
        redacted: redacted,
        uploadDate: uploadDate,
        uploadedBy: uploadedBy
      })
    } catch (err) {
      console.error('[DEBUG] View error full details:', {
        message: err.message,
        status: err?.response?.status,
        data: err?.response?.data
      })
      toast.error('Failed to load image preview')
      setPreviewModal(false)
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleResult = (data, uploadedFile) => {
    setResult(data)
    setFile(uploadedFile)
    loadStats()
    loadRecentLogs()
    toast.success('Image anonymized successfully!')
    setTimeout(() => {
      document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  const handleDownload = () => {
    if (result?.download_url) {
      window.open(result.download_url, '_blank')
    }
  }

  return (
    <div className="dashboard">
      <div className="container">
        <div className="welcome-section">
          <h1>
            Welcome, {user?.name}!
            <span className="badge badge-medical" style={{ marginLeft: '12px' }}>Medical Staff</span>
          </h1>
          <p>Upload medical images for anonymization</p>
        </div>

        <div className="tabs">
          <button
            onClick={() => setActiveTab('upload')}
            className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          >
            Upload & Anonymize
          </button>
          <button
            onClick={() => setActiveTab('all-images')}
            className={`tab ${activeTab === 'all-images' ? 'active' : ''}`}
          >
            All Anonymized Images
          </button>
        </div>

        {activeTab === 'upload' && (
          <>
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

            <UploadZone onResult={handleResult} onFileSelect={setFile} />

        {result && file && (
          <div id="result-section">
            <div className="card" style={{
              marginBottom: '2rem'
            }}>
              <h2 style={{ color: '#27ae60', marginBottom: '1.5rem' }}>
                ✓ Anonymization Complete
              </h2>
              
              <ResultViewer result={result} originalFile={file} />
              
              <StatsPanel log={{
                classification: result.classification,
                format: result.format,
                confidence: result.confidence,
                paddleRegions: result.paddle_regions,
                easyRegions: result.easy_regions,
                totalRegions: result.total_regions,
                redacted: result.redacted,
                skipped: result.skipped,
                processingTime: result.processingTime,
                isDicom: result.format === 'DICOM',
                tagsAnonymized: result.tags_anonymized
              }} />
              
              <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                <button onClick={handleDownload} className="btn-primary">
                  ⬇ Download Anonymized Image
                </button>
                <button
                  onClick={() => navigate(`/result/${result.logId}`)}
                  style={{
                    background: 'transparent',
                    border: '1px solid #00a8e8',
                    color: '#00a8e8',
                    padding: '0.75rem 1.5rem',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '1rem',
                    fontWeight: '500'
                  }}
                >
                  View Full Details →
                </button>
              </div>
            </div>
          </div>
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
                      <th>Regions</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentLogs.map(log => (
                      <tr 
                        key={log._id}
                        onClick={() => navigate(`/result/${log._id}`)}
                        style={{ cursor: 'pointer' }}
                      >
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
                        <td>{log.redacted || 0}</td>
                        <td>
                          <span className={`status-badge ${log.status}`}>
                            {log.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ textAlign: 'center', marginTop: '1rem' }}>
                  <button
                    onClick={() => navigate('/history')}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#00a8e8',
                      cursor: 'pointer',
                      fontSize: '1rem',
                      textDecoration: 'underline'
                    }}
                  >
                    View All History →
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'all-images' && (
          <div className="card">
            <h2 style={{ marginBottom: '1.5rem' }}>All Anonymized Images</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              View and download anonymized images from all users in the system.
            </p>

            {imagesLoading ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: '#00a8e8' }}>
                Loading images...
              </div>
            ) : allImages.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                No anonymized images found.
              </div>
            ) : (
              <table className="log-table">
                <thead>
                  <tr>
                    <th>Uploaded By</th>
                    <th>Date</th>
                    <th>Filename</th>
                    <th>Classification</th>
                    <th>Regions</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {allImages.map(img => (
                    <tr key={img.image_id}>
                      <td style={{ fontWeight: '500' }}>{img.uploaded_by}</td>
                      <td>{new Date(img.upload_date).toLocaleDateString()}</td>
                      <td>{img.filename}</td>
                      <td style={{
                        maxWidth: '150px',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {getClassificationLabel(img.classification)}
                      </td>
                      <td>{img.redacted || 0}</td>
                      <td>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            onClick={() => handleViewImage(img.image_id, img.filename, img.classification, img.redacted, img.upload_date, img.uploaded_by)}
                            className="btn-secondary"
                            style={{ padding: '0.4rem 0.8rem', fontSize: '12px' }}
                          >
                            View
                          </button>
                          <button
                            onClick={() => handleImageDownload(img.image_id, img.filename)}
                            className="btn-primary"
                            style={{ padding: '0.4rem 0.8rem', fontSize: '12px' }}
                          >
                            ⬇ Download
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {previewModal && (
          <div className="modal-overlay">
            <div className="modal-content" style={{ maxWidth: '900px' }}>
              <div className="modal-header">
                <h3 style={{ margin: 0 }}>
                  {previewImage?.filename || 'Image Preview'}
                </h3>
                <button
                  onClick={() => { setPreviewModal(false); setPreviewImage(null); }}
                  className="modal-close"
                >
                  ✕
                </button>
              </div>

              {previewLoading ? (
                <div style={{ textAlign: 'center', padding: '3rem', color: '#00a8e8' }}>
                  Loading image...
                </div>
              ) : previewImage?.url ? (
                <>
                  <div style={{
                    display: 'flex',
                    gap: '1rem',
                    marginBottom: '1rem',
                    fontSize: '0.9rem',
                    color: 'var(--text-secondary)',
                    flexWrap: 'wrap'
                  }}>
                    <span><strong>Uploaded by:</strong> {previewImage.uploadedBy}</span>
                    <span><strong>Classification:</strong> {getClassificationLabel(previewImage.classification)}</span>
                    <span><strong>Regions:</strong> {previewImage.redacted || 0}</span>
                    <span><strong>Date:</strong> {new Date(previewImage.uploadDate).toLocaleDateString()}</span>
                  </div>

                  <div style={{
                    background: '#f5f5f5',
                    borderRadius: '4px',
                    padding: '1rem',
                    textAlign: 'center'
                  }}>
                    <img
                      src={previewImage.url}
                      alt={previewImage.filename}
                      style={{
                        maxWidth: '100%',
                        maxHeight: '60vh',
                        objectFit: 'contain'
                      }}
                    />
                  </div>

                  <div style={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                    gap: '1rem',
                    marginTop: '1.5rem'
                  }}>
                    <button
                      onClick={() => { setPreviewModal(false); setPreviewImage(null); }}
                      className="btn-ghost"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => handleImageDownload(previewImage.imageId, previewImage.filename)}
                      className="btn-primary"
                    >
                      ⬇ Download
                    </button>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem', color: '#e74c3c' }}>
                  Failed to load image
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default MedicalDashboard
