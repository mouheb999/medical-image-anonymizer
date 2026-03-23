import { getClassificationLabel } from '../utils/classificationLabel'

const StatsPanel = ({ log }) => {
  return (
    <div className="stats-panel">
      <h2>Processing Statistics</h2>
      
      <div className="stats-grid-detail">
        <div className="stat-card-detail">
          <div className="stat-label">Classification</div>
          <div className="stat-value">
            {getClassificationLabel(log.classification)}
          </div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">Format</div>
          <div className="stat-value">{log.format || 'N/A'}</div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">Confidence</div>
          <div className="stat-value">
            {log.confidence ? `${(log.confidence * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">PaddleOCR Regions</div>
          <div className="stat-value">{log.paddleRegions || 0}</div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">EasyOCR Regions</div>
          <div className="stat-value">{log.easyRegions || 0}</div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">Total Merged</div>
          <div className="stat-value">{log.totalRegions || 0}</div>
        </div>
        
        <div className="stat-card-detail success">
          <div className="stat-label">Redacted</div>
          <div className="stat-value">{log.redacted || 0}</div>
        </div>
        
        <div className="stat-card-detail failed">
          <div className="stat-label">Skipped</div>
          <div className="stat-value">{log.skipped || 0}</div>
        </div>
        
        <div className="stat-card-detail">
          <div className="stat-label">Processing Time</div>
          <div className="stat-value">
            {log.processingTime ? `${(log.processingTime / 1000).toFixed(2)}s` : 'N/A'}
          </div>
        </div>
      </div>
      
      {log.isDicom && (
        <div className="dicom-info">
          <strong>DICOM Metadata:</strong> {log.tagsAnonymized || 0} tags anonymized
        </div>
      )}
    </div>
  )
}

export default StatsPanel
