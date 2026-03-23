import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import API from '../api/axios'
import LogTable from '../components/LogTable'

const History = () => {
  const { user } = useAuth()
  const [logs, setLogs] = useState([])
  const [pagination, setPagination] = useState(null)
  const [page, setPage] = useState(1)
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    loadLogs()
  }, [page, filter])

  const loadLogs = async () => {
    setLoading(true)
    try {
      const res = await API.get(`/history?page=${page}&limit=10`)
      let filteredLogs = res.data.data
      
      if (filter !== 'all') {
        filteredLogs = filteredLogs.filter(log => log.status === filter)
      }
      
      setLogs(filteredLogs)
      setPagination(res.data.pagination)
    } catch (err) {
      console.error('Failed to load history:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRowClick = (logId) => {
    navigate(`/result/${logId}`)
  }

  return (
    <div className="history-page">
      <div className="container">
        <div className="page-header">
          <h1>Anonymization History</h1>
          <div className="filter-group">
            <label>Filter:</label>
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
              <option value="pending">Pending</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <>
            <LogTable logs={logs} onRowClick={handleRowClick} isAdmin={false} />
            
            {pagination && pagination.pages > 1 && (
              <div className="pagination">
                <button 
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Previous
                </button>
                <span>Page {page} of {pagination.pages}</span>
                <button 
                  onClick={() => setPage(p => Math.min(pagination.pages, p + 1))}
                  disabled={page === pagination.pages}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default History
