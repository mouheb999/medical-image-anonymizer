import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import API from '../api/axios'

const Register = () => {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [role, setRole] = useState('')
  const [adminKey, setAdminKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleNext = (e) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    setStep(2)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (!role) {
      setError('Please select a role')
      return
    }

    setLoading(true)

    try {
      const res = await API.post('/auth/register', {
        name,
        email,
        password,
        role,
        adminKey: role === 'responsable' ? adminKey : undefined
      })
      localStorage.setItem('token', res.data.token)
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card" style={{ maxWidth: step === 2 ? '900px' : '450px' }}>
        <div className="auth-header">
          <h1>MedSecure</h1>
          <p>Medical Image Anonymization Platform</p>
        </div>
        
        <div className="auth-form">
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{
                width: '30px',
                height: '30px',
                borderRadius: '50%',
                background: step >= 1 ? 'var(--accent-cyan)' : 'var(--border-default)',
                color: '#080C14',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold'
              }}>1</div>
              <div style={{
                width: '60px',
                height: '2px',
                background: step >= 2 ? 'var(--accent-cyan)' : 'var(--border-default)'
              }}></div>
              <div style={{
                width: '30px',
                height: '30px',
                borderRadius: '50%',
                background: step >= 2 ? 'var(--accent-cyan)' : 'var(--border-default)',
                color: '#080C14',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold'
              }}>2</div>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}

          {step === 1 && (
            <form onSubmit={handleNext}>
              <h2>Basic Information</h2>
              
              <div className="form-group">
                <label htmlFor="name">Full Name</label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  placeholder="Enter your name"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="Enter your email"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="At least 6 characters"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password</label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  placeholder="Re-enter password"
                />
              </div>
              
              <button type="submit" className="btn-primary">
                Next →
              </button>
              
              <p className="auth-link">
                Already have an account? <Link to="/login">Login here</Link>
              </p>
            </form>
          )}

          {step === 2 && (
            <form onSubmit={handleSubmit}>
              <h2>Choose Your Role</h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div
                  onClick={() => setRole('utilisateur')}
                  style={{
                    border: role === 'utilisateur' ? '2px solid #00a8e8' : '1px solid #ddd',
                    background: role === 'utilisateur' ? '#e8f4fd' : 'white',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  <div style={{ fontSize: '2rem', textAlign: 'center', marginBottom: '0.5rem' }}>👤</div>
                  <h3 style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '1.1rem' }}>Basic User</h3>
                  <p style={{ textAlign: 'center', color: '#666', fontSize: '0.85rem', marginBottom: '1rem' }}>
                    Upload and anonymize medical images
                  </p>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '0.85rem' }}>
                    <li style={{ color: '#27ae60', marginBottom: '0.3rem' }}>✓ Upload images</li>
                    <li style={{ color: '#27ae60', marginBottom: '0.3rem' }}>✓ View anonymization result</li>
                    <li style={{ color: '#e74c3c', marginBottom: '0.3rem' }}>✗ Download images</li>
                    <li style={{ color: '#e74c3c' }}>✗ View history</li>
                  </ul>
                </div>

                <div
                  onClick={() => setRole('utilisateur_medical')}
                  style={{
                    border: role === 'utilisateur_medical' ? '2px solid #00a8e8' : '1px solid #ddd',
                    background: role === 'utilisateur_medical' ? '#e8f4fd' : 'white',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                    position: 'relative'
                  }}
                >
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'var(--success)',
                    color: '#080C14',
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '10px',
                    fontWeight: '600'
                  }}>Recommended</div>
                  <div style={{ fontSize: '2rem', textAlign: 'center', marginBottom: '0.5rem' }}>🏥</div>
                  <h3 style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '1.1rem' }}>Medical Staff</h3>
                  <p style={{ textAlign: 'center', color: '#666', fontSize: '0.85rem', marginBottom: '1rem' }}>
                    Full medical workflow access
                  </p>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '0.85rem' }}>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ Upload images</li>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ Download anonymized images</li>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ View history</li>
                    <li style={{ color: 'var(--success)' }}>✓ Before/after comparison</li>
                  </ul>
                </div>

                <div
                  onClick={() => setRole('responsable')}
                  style={{
                    border: role === 'responsable' ? '2px solid #00a8e8' : '1px solid #ddd',
                    background: role === 'responsable' ? '#e8f4fd' : 'white',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  <div style={{ fontSize: '2rem', textAlign: 'center', marginBottom: '0.5rem' }}>🛡️</div>
                  <h3 style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '1.1rem' }}>Administrator</h3>
                  <p style={{ textAlign: 'center', color: '#666', fontSize: '0.85rem', marginBottom: '1rem' }}>
                    System administration
                  </p>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '0.85rem' }}>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ Manage users</li>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ View all logs</li>
                    <li style={{ color: 'var(--success)', marginBottom: '0.3rem' }}>✓ Supervise storage</li>
                    <li style={{ color: 'var(--success)' }}>✓ Configure settings</li>
                  </ul>
                </div>
              </div>

              {role === 'responsable' && (
                <div className="form-group">
                  <label htmlFor="adminKey">Admin Registration Key</label>
                  <input
                    id="adminKey"
                    type="password"
                    value={adminKey}
                    onChange={(e) => setAdminKey(e.target.value)}
                    required
                    placeholder="Key provided by Pura Solutions"
                  />
                </div>
              )}

              <div style={{ display: 'flex', gap: '1rem' }}>
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="btn-secondary"
                  style={{ flex: 1 }}
                >
                  ← Back
                </button>
                <button type="submit" disabled={loading} className="btn-primary" style={{ flex: 1 }}>
                  {loading ? 'Creating account...' : 'Create Account'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}

export default Register
