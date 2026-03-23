import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import API from '../api/axios'
import { getClassificationLabel } from '../utils/classificationLabel'

const AdminDashboard = () => {
  const { user } = useAuth()
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState('overview')
  const [stats, setStats] = useState(null)
  const [users, setUsers] = useState([])
  const [logs, setLogs] = useState([])
  const [pagination, setPagination] = useState(null)
  const [settings, setSettings] = useState(null)
  const [statusFilter, setStatusFilter] = useState('all')
  const [searchFilter, setSearchFilter] = useState('')
  const [page, setPage] = useState(1)
  const [saved, setSaved] = useState(false)
  const [userSearch, setUserSearch] = useState('')
  const [selectedUser, setSelectedUser] = useState(null)
  const [userImages, setUserImages] = useState([])
  const [imagesLoading, setImagesLoading] = useState(false)
  const [previewModal, setPreviewModal] = useState(false)
  const [previewImage, setPreviewImage] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  useEffect(() => {
    if (activeTab === 'overview') loadStats()
    if (activeTab === 'users') loadUsers()
    if (activeTab === 'logs') loadLogs()
    if (activeTab === 'settings') loadSettings()
  }, [activeTab, page, statusFilter, searchFilter])

  const loadStats = async () => {
    try {
      const res = await API.get('/history/admin/stats')
      setStats(res.data.data)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const loadUsers = async () => {
    try {
      const res = await API.get('/history/admin/users')
      setUsers(res.data.data)
    } catch (err) {
      console.error('Failed to load users:', err)
    }
  }

  const loadLogs = async () => {
    try {
      const res = await API.get(`/history/admin/logs?page=${page}&status=${statusFilter}&search=${searchFilter}`)
      setLogs(res.data.data)
      setPagination(res.data.pagination)
    } catch (err) {
      console.error('Failed to load logs:', err)
    }
  }

  const loadSettings = async () => {
    try {
      const res = await API.get('/history/admin/settings')
      setSettings(res.data.data)
    } catch (err) {
      console.error('Failed to load settings:', err)
    }
  }

  const deleteUser = async (userId, userName) => {
    if (!window.confirm(`Are you sure you want to delete ${userName}? This action is irreversible and will delete all their images.`)) {
      return
    }
    try {
      const res = await API.delete(`/history/admin/users/${userId}`)
      toast.success(`User deleted. ${res.data.deleted_images_count} images removed.`)
      loadUsers()
    } catch (err) {
      toast.error('Failed to delete user')
    }
  }

  const loadUserImages = async (userId) => {
    setImagesLoading(true)
    try {
      const res = await API.get(`/history/admin/users/${userId}/images`)
      setSelectedUser(res.data.user)
      setUserImages(res.data.data)
    } catch (err) {
      console.error('Failed to load user images:', err)
    } finally {
      setImagesLoading(false)
    }
  }

  const deleteImage = async (imageId) => {
    if (!window.confirm('Are you sure you want to delete this image? This action is irreversible.')) {
      return
    }
    try {
      await API.delete(`/history/admin/images/${imageId}`)
      setUserImages(prev => prev.filter(img => img.image_id !== imageId))
    } catch (err) {
      toast.error('Failed to delete image')
    }
  }

  const handleImageDownload = async (imageId, filename) => {
    try {
      console.log('[DEBUG] Download requested for imageId:', imageId)
      console.log('[DEBUG] Token in localStorage:', localStorage.getItem('token') ? 'EXISTS' : 'MISSING')
      
      const res = await API.get(`/history/admin/images/${imageId}/download`)
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

  const handleViewImage = async (imageId, filename, classification, redacted, uploadDate) => {
    setPreviewLoading(true)
    setPreviewModal(true)
    try {
      console.log('[DEBUG] View requested for imageId:', imageId)
      const res = await API.get(`/history/admin/images/${imageId}/download`)
      console.log('[DEBUG] View response:', res.data)
      setPreviewImage({
        url: res.data.presigned_url,
        filename: filename,
        imageId: imageId,
        classification: classification,
        redacted: redacted,
        uploadDate: uploadDate
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

  const changeRole = async (userId, newRole) => {
    try {
      await API.put(`/history/admin/users/${userId}/role`, { role: newRole })
      loadUsers()
    } catch (err) {
      toast.error('Failed to change user role')
    }
  }

  const saveSettings = async () => {
    try {
      await API.put('/history/admin/settings', settings)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (err) {
      toast.error('Failed to save settings')
    }
  }

  const filteredUsers = users.filter(u => 
    u.name.toLowerCase().includes(userSearch.toLowerCase()) ||
    u.email.toLowerCase().includes(userSearch.toLowerCase())
  )

  return (
    <div className="dashboard">
      <div className="container">
        <div className="welcome-section">
          <h1>
            Admin Panel
            <span className="badge badge-responsable" style={{ marginLeft: '12px' }}>Administrator</span>
          </h1>
          <p>System administration and monitoring</p>
        </div>

        <div className="card" style={{ marginBottom: '2rem' }}>
          <div className="tabs">
            {['overview', 'users', 'logs', 'settings'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`tab ${activeTab === tab ? 'active' : ''}`}
                style={{ textTransform: 'capitalize' }}
              >
                {tab}
              </button>
            ))}
          </div>

          <div style={{ padding: '1rem 0' }}>
            {activeTab === 'overview' && stats && (
              <>
                <div className="stats-grid" style={{ marginBottom: '2rem' }}>
                  <div className="stat-card">
                    <div className="stat-value">{stats.users.total}</div>
                    <div className="stat-label">Total Users</div>
                  </div>
                  <div className="stat-card success">
                    <div className="stat-value">{stats.images.success}</div>
                    <div className="stat-label">Successful</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{stats.images.total}</div>
                    <div className="stat-label">Total Images</div>
                  </div>
                  <div className="stat-card success">
                    <div className="stat-value">
                      {stats.images.total > 0 
                        ? Math.round((stats.images.success / stats.images.total) * 100)
                        : 0}%
                    </div>
                    <div className="stat-label">Success Rate</div>
                  </div>
                </div>

                <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Storage by Category</h3>
                <div style={{ marginBottom: '2rem' }}>
                  {stats.storageByCategory.map(cat => {
                    const total = stats.storageByCategory.reduce((sum, c) => sum + c.count, 0)
                    const percentage = (cat.count / total) * 100
                    return (
                      <div key={cat._id} style={{ marginBottom: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                          <span style={{ fontWeight: '500' }}>{cat._id}</span>
                          <span style={{ color: 'var(--text-secondary)' }}>{cat.count} images</span>
                        </div>
                        <div style={{
                          width: '100%',
                          height: '8px',
                          background: '#e8f4fd',
                          borderRadius: '4px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${percentage}%`,
                            height: '100%',
                            background: '#00a8e8',
                            transition: 'width 0.3s'
                          }}></div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Recent Activity</h3>
                <table className="log-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>User</th>
                      <th>File</th>
                      <th>Classification</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.recentActivity.map(log => (
                      <tr key={log._id}>
                        <td>{new Date(log.createdAt).toLocaleDateString()}</td>
                        <td>{log.user?.name || 'N/A'}</td>
                        <td>{log.originalFilename}</td>
                        <td>{getClassificationLabel(log.classification)}</td>
                        <td>
                          <span className={`status-badge ${log.status}`}>
                            {log.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {activeTab === 'users' && !selectedUser && (
              <>
                <div style={{ marginBottom: '1.5rem' }}>
                  <input
                    type="text"
                    placeholder="Search by name or email..."
                    value={userSearch}
                    onChange={(e) => setUserSearch(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      fontSize: '1rem'
                    }}
                  />
                </div>

                <table className="log-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Email</th>
                      <th>Role</th>
                      <th>Images</th>
                      <th>Joined</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.map(u => (
                      <tr 
                        key={u._id}
                        onClick={() => loadUserImages(u._id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td>{u.name}</td>
                        <td>{u.email}</td>
                        <td>
                          <span className={`badge ${u.role === 'responsable' ? 'badge-responsable' : u.role === 'utilisateur_medical' ? 'badge-medical' : 'badge-utilisateur'}`}>
                            {u.role === 'responsable' ? 'Responsable' : u.role === 'utilisateur_medical' ? 'Medical' : 'User'}
                          </span>
                        </td>
                        <td>{u.totalImages || 0}</td>
                        <td>{new Date(u.createdAt).toLocaleDateString()}</td>
                        <td>
                          <div style={{ display: 'flex', gap: '0.5rem' }} onClick={(e) => e.stopPropagation()}>
                            <button
                              onClick={() => deleteUser(u._id, u.name)}
                              disabled={u.role === 'responsable'}
                              className="btn-danger"
                              style={{ padding: '0.4rem 0.8rem', fontSize: '12px' }}
                            >
                              Delete
                            </button>
                            <select
                              value={u.role}
                              onChange={(e) => changeRole(u._id, e.target.value)}
                              className="form-group"
                              style={{ padding: '0.4rem', fontSize: '12px', margin: 0 }}
                            >
                              <option value="utilisateur">User</option>
                              <option value="utilisateur_medical">Medical</option>
                              <option value="responsable">Responsable</option>
                            </select>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {activeTab === 'users' && selectedUser && (
              <>
                <div style={{ marginBottom: '1.5rem' }}>
                  <button
                    onClick={() => { setSelectedUser(null); setUserImages([]); }}
                    style={{
                      background: 'transparent',
                      border: '1px solid #00a8e8',
                      color: '#00a8e8',
                      padding: '0.5rem 1rem',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      marginBottom: '1rem'
                    }}
                  >
                    ← Back to Users
                  </button>
                  <h3 style={{ color: 'var(--text-primary)' }}>
                    Images for {selectedUser.name} ({selectedUser.email})
                  </h3>
                </div>

                {imagesLoading ? (
                  <div style={{ textAlign: 'center', padding: '2rem', color: '#00a8e8' }}>
                    Loading images...
                  </div>
                ) : userImages.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                    No images found for this user.
                  </div>
                ) : (
                  <table className="log-table">
                    <thead>
                      <tr>
                        <th>Filename</th>
                        <th>Upload Date</th>
                        <th>Classification</th>
                        <th>Regions</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {userImages.map(img => (
                        <tr key={img.image_id}>
                          <td>{img.filename}</td>
                          <td>{new Date(img.upload_date).toLocaleDateString()}</td>
                          <td>{getClassificationLabel(img.classification)}</td>
                          <td>{img.redacted || 0}</td>
                          <td>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                              <button
                                onClick={() => handleViewImage(img.image_id, img.filename, img.classification, img.redacted, img.upload_date)}
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
                                Download
                              </button>
                              <button
                                onClick={() => deleteImage(img.image_id)}
                                className="btn-danger"
                                style={{ padding: '0.4rem 0.8rem', fontSize: '12px' }}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            )}

            {activeTab === 'logs' && (
              <>
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                  <select
                    value={statusFilter}
                    onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
                    style={{
                      padding: '0.75rem',
                      borderRadius: '4px',
                      border: '1px solid #ddd',
                      fontSize: '1rem'
                    }}
                  >
                    <option value="all">All Status</option>
                    <option value="success">Success</option>
                    <option value="failed">Failed</option>
                    <option value="pending">Pending</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Search by filename..."
                    value={searchFilter}
                    onChange={(e) => { setSearchFilter(e.target.value); setPage(1); }}
                    style={{
                      flex: 1,
                      padding: '0.75rem',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      fontSize: '1rem'
                    }}
                  />
                </div>

                <table className="log-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>User</th>
                      <th>File</th>
                      <th>Classification</th>
                      <th>Regions</th>
                      <th>Time</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map(log => (
                      <tr key={log._id}>
                        <td>{new Date(log.createdAt).toLocaleDateString()}</td>
                        <td>{log.user?.name || 'N/A'}</td>
                        <td>{log.originalFilename}</td>
                        <td>{getClassificationLabel(log.classification)}</td>
                        <td>{log.redacted || 0}</td>
                        <td>
                          {log.processingTime 
                            ? `${(log.processingTime / 1000).toFixed(2)}s`
                            : 'N/A'}
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

                {pagination && pagination.pages > 1 && (
                  <div className="pagination">
                    <button 
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      ← Previous
                    </button>
                    <span>Page {page} of {pagination.pages}</span>
                    <button 
                      onClick={() => setPage(p => Math.min(pagination.pages, p + 1))}
                      disabled={page === pagination.pages}
                    >
                      Next →
                    </button>
                  </div>
                )}
              </>
            )}

            {activeTab === 'settings' && settings && (
              <>
                <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>OCR Configuration</h3>
                <div style={{ marginBottom: '2rem' }}>
                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                      Confidence Threshold: {settings.ocrConfidenceThreshold}
                    </label>
                    <input
                      type="range"
                      min="0.05"
                      max="1"
                      step="0.05"
                      value={settings.ocrConfidenceThreshold}
                      onChange={(e) => setSettings({...settings, ocrConfidenceThreshold: parseFloat(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                      Border Margin (px)
                    </label>
                    <input
                      type="number"
                      min="50"
                      max="300"
                      value={settings.borderMarginPx}
                      onChange={(e) => setSettings({...settings, borderMarginPx: parseInt(e.target.value)})}
                      style={{
                        padding: '0.5rem',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        width: '200px'
                      }}
                    />
                  </div>

                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                      IoU Threshold: {settings.iouThreshold}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={settings.iouThreshold}
                      onChange={(e) => setSettings({...settings, iouThreshold: parseFloat(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                      Inpaint Radius
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={settings.inpaintRadius}
                      onChange={(e) => setSettings({...settings, inpaintRadius: parseInt(e.target.value)})}
                      style={{
                        padding: '0.5rem',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        width: '200px'
                      }}
                    />
                  </div>
                </div>

                <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Pipeline Toggles</h3>
                <div style={{ marginBottom: '2rem' }}>
                  <label style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={settings.paddleOcrEnabled}
                      onChange={(e) => setSettings({...settings, paddleOcrEnabled: e.target.checked})}
                      style={{ marginRight: '0.5rem', width: '20px', height: '20px' }}
                    />
                    <span style={{ fontWeight: '500' }}>Enable PaddleOCR</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={settings.easyOcrEnabled}
                      onChange={(e) => setSettings({...settings, easyOcrEnabled: e.target.checked})}
                      style={{ marginRight: '0.5rem', width: '20px', height: '20px' }}
                    />
                    <span style={{ fontWeight: '500' }}>Enable EasyOCR</span>
                  </label>
                </div>

                <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>DICOM PHI Tags</h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '0.5rem',
                  marginBottom: '2rem'
                }}>
                  {settings.dicomTagsToRemove.map(tag => (
                    <label key={tag} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={true}
                        readOnly
                        style={{ marginRight: '0.5rem' }}
                      />
                      <span style={{ fontSize: '0.9rem' }}>{tag}</span>
                    </label>
                  ))}
                </div>

                <button
                  onClick={saveSettings}
                  className="btn-primary"
                  style={{ marginRight: '1rem' }}
                >
                  Save Settings
                </button>

                {saved && (
                  <span style={{
                    color: '#27ae60',
                    fontWeight: '500',
                    fontSize: '1rem'
                  }}>
                    ✓ Settings saved successfully
                  </span>
                )}
              </>
            )}
          </div>
        </div>

        {previewModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div className="modal-content" style={{
              maxWidth: '900px',
              width: '90%',
              maxHeight: '90vh',
              overflow: 'auto'
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1rem'
              }}>
                <h3 style={{ margin: 0, color: 'var(--text-primary)' }}>
                  {previewImage?.filename || 'Image Preview'}
                </h3>
                <button
                  onClick={() => { setPreviewModal(false); setPreviewImage(null); }}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    fontSize: '1.5rem',
                    cursor: 'pointer',
                    color: 'var(--text-secondary)'
                  }}
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
                    color: 'var(--text-secondary)'
                  }}>
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
                      style={{
                        background: 'transparent',
                        border: '1px solid #ddd',
                        color: 'var(--text-secondary)',
                        padding: '0.75rem 1.5rem',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
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

export default AdminDashboard
