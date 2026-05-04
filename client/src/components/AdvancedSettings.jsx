import { useState } from 'react'
import '../styles/advanced-settings.css'

const AdvancedSettings = ({ settings, onChange }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  const handleChange = (key, value) => {
    onChange({ ...settings, [key]: value })
  }

  return (
    <div className="advanced-settings">
      <button
        className="btn-ghost"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{ marginBottom: '1rem', width: '100%' }}
      >
        {isExpanded ? '▼' : '▶'} Advanced Settings (OCR & Redaction Parameters)
      </button>

      {isExpanded && (
        <div className="settings-panel">
          <div className="settings-grid">
            {/* OCR Confidence Threshold */}
            <div className="setting-item">
              <label htmlFor="conf_threshold">
                OCR Confidence Threshold
                <span className="setting-hint">Min confidence to keep detection (0.0-1.0)</span>
              </label>
              <div className="slider-container">
                <input
                  id="conf_threshold"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.conf_threshold}
                  onChange={(e) => handleChange('conf_threshold', parseFloat(e.target.value))}
                />
                <span className="slider-value">{settings.conf_threshold.toFixed(1)}</span>
              </div>
              <p className="setting-description">
                Lower = more detections (may include false positives)<br/>
                Higher = fewer detections (may miss faint text)
              </p>
            </div>

            {/* Padding */}
            <div className="setting-item">
              <label htmlFor="padding">
                Redaction Padding (pixels)
                <span className="setting-hint">Extra pixels around detected text</span>
              </label>
              <div className="slider-container">
                <input
                  id="padding"
                  type="range"
                  min="0"
                  max="20"
                  step="1"
                  value={settings.padding}
                  onChange={(e) => handleChange('padding', parseInt(e.target.value))}
                />
                <span className="slider-value">{settings.padding}px</span>
              </div>
              <p className="setting-description">
                Ensures complete text removal by expanding redaction area
              </p>
            </div>

            {/* Border Margin */}
            <div className="setting-item">
              <label htmlFor="border_margin">
                Border Safety Margin (pixels)
                <span className="setting-hint">Only redact text within this margin from edges</span>
              </label>
              <div className="slider-container">
                <input
                  id="border_margin"
                  type="range"
                  min="50"
                  max="200"
                  step="10"
                  value={settings.border_margin}
                  onChange={(e) => handleChange('border_margin', parseInt(e.target.value))}
                />
                <span className="slider-value">{settings.border_margin}px</span>
              </div>
              <p className="setting-description">
                Protects diagnostic content in center by only redacting borders
              </p>
            </div>

            {/* Border Percentage (EasyOCR) */}
            <div className="setting-item">
              <label htmlFor="border_pct">
                Border Scan Percentage
                <span className="setting-hint">How much of border to scan with EasyOCR (0.0-1.0)</span>
              </label>
              <div className="slider-container">
                <input
                  id="border_pct"
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={settings.border_pct}
                  onChange={(e) => handleChange('border_pct', parseFloat(e.target.value))}
                />
                <span className="slider-value">{(settings.border_pct * 100).toFixed(0)}%</span>
              </div>
              <p className="setting-description">
                Larger = more area scanned = slower but more thorough
              </p>
            </div>
          </div>

          <div className="settings-info">
            <p><strong>ℹ️ How these parameters work:</strong></p>
            <ul>
              <li><strong>Confidence Threshold:</strong> Filters OCR detections - only boxes with confidence ≥ threshold are kept</li>
              <li><strong>Padding:</strong> Expands redaction boxes to ensure complete text removal</li>
              <li><strong>Border Margin:</strong> Safety zone - text in center is never auto-redacted</li>
              <li><strong>Border Scan:</strong> Controls EasyOCR scan area (performance vs thoroughness)</li>
            </ul>
          </div>

          <button
            className="btn-secondary"
            onClick={() => onChange({
              conf_threshold: 0.1,
              padding: 5,
              border_margin: 100,
              border_pct: 0.20
            })}
            style={{ marginTop: '1rem' }}
          >
            Reset to Defaults
          </button>
        </div>
      )}
    </div>
  )
}

export default AdvancedSettings
