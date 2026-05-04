import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import { useState, useEffect } from 'react'

const Navbar = () => {
  const { user, logout } = useAuth()
  const { theme, toggleTheme } = useTheme()
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const getRoleBadgeClass = (role) => {
    switch (role) {
      case 'responsable': return 'badge badge-responsable'
      case 'utilisateur_medical': return 'badge badge-medical'
      default: return 'badge badge-utilisateur'
    }
  }

  const getRoleLabel = (role) => {
    switch (role) {
      case 'responsable': return 'Admin'
      case 'utilisateur_medical': return 'Medical'
      default: return 'User'
    }
  }

  const isActive = (path) => location.pathname === path

  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="nav-container">
        <Link to="/dashboard" className="nav-logo">
          MedSecure
          <span className="status-dot" />
        </Link>
        
        <div className="nav-links">
          <Link to="/dashboard" className={isActive('/dashboard') ? 'active' : ''}>
            Dashboard
          </Link>
          <Link to="/pathology" className={isActive('/pathology') ? 'active' : ''}>
            Pathology
          </Link>
          {['utilisateur_medical', 'responsable'].includes(user?.role) && (
            <Link to="/history" className={isActive('/history') ? 'active' : ''}>
              History
            </Link>
          )}
          {user?.role === 'responsable' && (
            <Link to="/admin" className={isActive('/admin') ? 'active' : ''}>
              Admin Panel
            </Link>
          )}
        </div>
        
        <div className="nav-user">
          <span className={getRoleBadgeClass(user?.role)}>
            {getRoleLabel(user?.role)}
          </span>
          <span className="user-name">{user?.name}</span>
          
          <button 
            onClick={toggleTheme} 
            className="theme-toggle"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5"/>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
          
          <button onClick={logout} className="btn-logout">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '6px' }}>
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            Logout
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
