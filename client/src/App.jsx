import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import { ToastProvider } from './context/ToastContext'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Result from './pages/Result'
import MedicalDashboard from './pages/MedicalDashboard'
import AdminDashboard from './pages/AdminDashboard'
import PathologyDetector from './pages/PathologyDetector'
import RoleDashboard from './components/RoleDashboard'
import Navbar from './components/Navbar'

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
      <ToastProvider>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<Navigate to="/dashboard" />} />
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <Navbar />
              <RoleDashboard />
            </ProtectedRoute>
          } />
          <Route path="/medical" element={
            <ProtectedRoute medicalOnly={true}>
              <Navbar />
              <MedicalDashboard />
            </ProtectedRoute>
          } />
          <Route path="/admin" element={
            <ProtectedRoute adminOnly={true}>
              <Navbar />
              <AdminDashboard />
            </ProtectedRoute>
          } />
          <Route path="/history" element={
            <ProtectedRoute medicalOnly={true}>
              <Navbar />
              <History />
            </ProtectedRoute>
          } />
          <Route path="/pathology" element={
            <ProtectedRoute>
              <Navbar />
              <PathologyDetector />
            </ProtectedRoute>
          } />
          <Route path="/result/:id" element={
            <ProtectedRoute>
              <Navbar />
              <Result />
            </ProtectedRoute>
          } />
        </Routes>
      </AuthProvider>
      </ToastProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}

export default App
