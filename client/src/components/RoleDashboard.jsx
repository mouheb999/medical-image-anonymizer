import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import Dashboard from '../pages/Dashboard'

const RoleDashboard = () => {
  const { user } = useAuth()
  
  if (!user) return <Navigate to="/login" />
  if (user.role === 'responsable') return <Navigate to="/admin" />
  if (user.role === 'utilisateur_medical') return <Navigate to="/medical" />
  
  return <Dashboard />
}

export default RoleDashboard
