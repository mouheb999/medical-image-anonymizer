require('dotenv').config()
const express = require('express')
const cors = require('cors')
const morgan = require('morgan')
const axios = require('axios')
const FormData = require('form-data')
const multer = require('multer')

const app = express()
const PORT = process.env.PORT || 5000

// In-memory storage for testing (no MongoDB)
const users = []
const logs = []
let userIdCounter = 1
let logIdCounter = 1

// Middleware
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(morgan('dev'))

// Multer setup
const storage = multer.memoryStorage()
const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }
})

// Mock JWT token generation
const generateToken = (userId) => {
  return `test-token-${userId}-${Date.now()}`
}

// Mock authentication middleware
const protect = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1]
  
  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'Not authorized — no token'
    })
  }
  
  const userId = token.split('-')[2]
  const user = users.find(u => u.id === parseInt(userId))
  
  if (!user) {
    return res.status(401).json({
      success: false,
      message: 'User not found'
    })
  }
  
  req.user = user
  next()
}

// Auth routes
app.post('/api/auth/register', (req, res) => {
  const { name, email, password } = req.body
  
  const existingUser = users.find(u => u.email === email)
  if (existingUser) {
    return res.status(400).json({
      success: false,
      message: 'Email already registered'
    })
  }
  
  const user = {
    id: userIdCounter++,
    name,
    email,
    role: 'user',
    createdAt: new Date()
  }
  
  users.push(user)
  const token = generateToken(user.id)
  
  res.status(201).json({
    success: true,
    token,
    user: {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role
    }
  })
})

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body
  
  const user = users.find(u => u.email === email)
  if (!user) {
    return res.status(401).json({
      success: false,
      message: 'Invalid email or password'
    })
  }
  
  const token = generateToken(user.id)
  
  res.json({
    success: true,
    token,
    user: {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role
    }
  })
})

app.get('/api/auth/me', protect, (req, res) => {
  res.json({
    success: true,
    user: {
      id: req.user.id,
      name: req.user.name,
      email: req.user.email,
      role: req.user.role
    }
  })
})

// Anonymize route
app.post('/api/anonymize', protect, upload.single('file'), async (req, res) => {
  const startTime = Date.now()
  
  const log = {
    id: logIdCounter++,
    userId: req.user.id,
    originalFilename: req.file.originalname,
    status: 'pending',
    createdAt: new Date()
  }
  
  logs.push(log)
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No file uploaded'
      })
    }
    
    const formData = new FormData()
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    })
    
    const fastApiResponse = await axios.post(
      `${process.env.FASTAPI_URL}/anonymize`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 300000
      }
    )
    
    const result = fastApiResponse.data
    const processingTime = Date.now() - startTime
    
    // Update log
    Object.assign(log, {
      anonymizedFilename: result.output_filename,
      classification: result.classification,
      confidence: result.confidence,
      format: result.format,
      isDicom: result.format === 'DICOM',
      tagsAnonymized: result.tags_anonymized || 0,
      paddleRegions: result.paddle_regions || 0,
      easyRegions: result.easy_regions || 0,
      totalRegions: result.total_regions || 0,
      redacted: result.redacted || 0,
      skipped: result.skipped || 0,
      minioUri: result.minio_uri,
      downloadUrl: result.download_url,
      processingTime,
      status: 'success'
    })
    
    res.json({
      success: true,
      logId: log.id,
      processingTime,
      ...result
    })
    
  } catch (error) {
    log.status = 'failed'
    log.errorMessage = error.response?.data?.detail || error.message
    
    const statusCode = error.response?.status || 500
    const message = error.response?.data?.detail || 
                    error.response?.data?.error ||
                    error.message
    
    res.status(statusCode).json({
      success: false,
      message: `Processing failed: ${message}` 
    })
  }
})

// History routes
app.get('/api/history', protect, (req, res) => {
  const userLogs = logs.filter(log => log.userId === req.user.id)
  
  res.json({
    success: true,
    data: userLogs.reverse(),
    pagination: {
      page: 1,
      limit: 10,
      total: userLogs.length,
      pages: 1
    }
  })
})

app.get('/api/history/:id', protect, (req, res) => {
  const log = logs.find(l => l.id === parseInt(req.params.id) && l.userId === req.user.id)
  
  if (!log) {
    return res.status(404).json({
      success: false,
      message: 'Log not found'
    })
  }
  
  res.json({ success: true, data: log })
})

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date(),
    mode: 'TEST MODE (No MongoDB)',
    users: users.length,
    logs: logs.length
  })
})

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ success: false, message: 'Route not found' })
})

// Error handler
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal server error'
  })
})

app.listen(PORT, () => {
  console.log(`
  ╔════════════════════════════════════════╗
  ║   Medical Anonymizer Backend           ║
  ║   🧪 TEST MODE (No MongoDB)            ║
  ║   Server running on port ${PORT}          ║
  ║   Users in memory: ${users.length}                  ║
  ║   Logs in memory: ${logs.length}                   ║
  ╚════════════════════════════════════════╝
  `)
})
