const jwt = require('jsonwebtoken')
const User = require('../models/User')

const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE
  })
}

const register = async (req, res) => {
  try {
    const { name, email, password, role, adminKey } = req.body

    if (role === 'responsable') {
      if (!adminKey || adminKey !== process.env.ADMIN_REGISTRATION_KEY) {
        return res.status(403).json({
          success: false,
          message: 'Invalid admin registration key'
        })
      }
    }

    const existingUser = await User.findOne({ email })
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered'
      })
    }

    const validRoles = ['utilisateur', 'utilisateur_medical', 'responsable']
    const userRole = validRoles.includes(role) ? role : 'utilisateur'

    const user = await User.create({ 
      name, email, password, role: userRole 
    })
    const token = generateToken(user._id)

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role
      }
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    })
  }
}

const login = async (req, res) => {
  try {
    const { email, password } = req.body
    
    const user = await User.findOne({ email }).select('+password')
    if (!user || !(await user.comparePassword(password))) {
      return res.status(401).json({
        success: false,
        message: 'Invalid email or password'
      })
    }
    
    const token = generateToken(user._id)
    
    res.json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role
      }
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    })
  }
}

const getMe = async (req, res) => {
  res.json({
    success: true,
    user: {
      id: req.user._id,
      name: req.user.name,
      email: req.user.email,
      role: req.user.role
    }
  })
}

module.exports = { register, login, getMe }
