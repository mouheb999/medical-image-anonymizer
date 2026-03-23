import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const ProtectedRoute = ({ 
  children, 
  adminOnly = false, 
  medicalOnly = false 
}) => {
  const { user, loading } = useAuth()

  if (loading) return (
    <div style={{ 
      display: 'flex', justifyContent: 'center', 
      alignItems: 'center', height: '100vh',
      color: '#00a8e8', fontSize: '18px'
    }}>
      Loading...
    </div>
  )

  if (!user) return <Navigate to="/login" />
  
  if (adminOnly && user.role !== 'responsable') 
    return <Navigate to="/dashboard" />
  
  if (medicalOnly && 
      !['utilisateur_medical', 'responsable'].includes(user.role)) 
    return <Navigate to="/dashboard" />

  return children
}

export default ProtectedRoute
