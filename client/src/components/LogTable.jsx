import { getClassificationLabel } from '../utils/classificationLabel'

const LogTable = ({ logs, onRowClick, isAdmin }) => {
  if (logs.length === 0) {
    return <div className="no-data">No logs found</div>
  }

  return (
    <table className="log-table">
      <thead>
        <tr>
          <th>Date</th>
          {isAdmin && <th>User</th>}
          <th>Original Filename</th>
          <th>Classification</th>
          <th>Regions Redacted</th>
          <th>Status</th>
          <th>Processing Time</th>
        </tr>
      </thead>
      <tbody>
        {logs.map(log => (
          <tr 
            key={log._id} 
            onClick={() => onRowClick && onRowClick(log._id)}
            className="clickable-row"
          >
            <td>{new Date(log.createdAt).toLocaleDateString()}</td>
            {isAdmin && (
              <td>{log.user?.name || log.user?.email || 'N/A'}</td>
            )}
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
            <td>
              {log.processingTime 
                ? `${(log.processingTime / 1000).toFixed(2)}s` 
                : 'N/A'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default LogTable
