const jwt = require('jsonwebtoken')
const User = require('../models/User')

const protect = async (req, res, next) => {
  try {
    let token
    
    if (req.headers.authorization?.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1]
    }
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Not authorized — no token'
      })
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    req.user = await User.findById(decoded.id)
    
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'User no longer exists'
      })
    }
    
    next()
  } catch (error) {
    return res.status(401).json({
      success: false,
      message: 'Not authorized — invalid token'
    })
  }
}

const responsableOnly = (req, res, next) => {
  if (req.user.role !== 'responsable') {
    return res.status(403).json({
      success: false,
      message: 'Responsable access required'
    })
  }
  next()
}

const medicalOrResponsable = (req, res, next) => {
  if (!['utilisateur_medical', 'responsable'].includes(req.user.role)) {
    return res.status(403).json({
      success: false,
      message: 'Medical user access required'
    })
  }
  next()
}

module.exports = { protect, responsableOnly, medicalOrResponsable }
