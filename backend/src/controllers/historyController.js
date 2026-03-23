const Log = require('../models/Log')
const User = require('../models/User')

const getHistory = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const skip = (page - 1) * limit
    
    const logs = await Log.find({ user: req.user._id })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .select('-__v')
    
    const total = await Log.countDocuments({ user: req.user._id })
    
    res.json({
      success: true,
      data: logs,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getLogById = async (req, res) => {
  try {
    const log = await Log.findOne({
      _id: req.params.id,
      user: req.user._id
    })
    
    if (!log) {
      return res.status(404).json({
        success: false,
        message: 'Log not found'
      })
    }
    
    res.json({ success: true, data: log })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getAllLogs = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 20
    const skip = (page - 1) * limit
    
    const logs = await Log.find()
      .populate('user', 'name email role')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
    
    const total = await Log.countDocuments()
    
    const stats = await Log.aggregate([
      {
        $group: {
          _id: null,
          totalImages: { $sum: 1 },
          successCount: {
            $sum: { $cond: [{ $eq: ['$status', 'success'] }, 1, 0] }
          },
          failedCount: {
            $sum: { $cond: [{ $eq: ['$status', 'failed'] }, 1, 0] }
          },
          avgProcessingTime: { $avg: '$processingTime' },
          totalRegionsRedacted: { $sum: '$redacted' }
        }
      }
    ])
    
    res.json({
      success: true,
      data: logs,
      stats: stats[0] || {},
      pagination: { page, limit, total, pages: Math.ceil(total / limit) }
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getAllUsers = async (req, res) => {
  try {
    const users = await User.find().select('-password')
    const usersWithStats = await Promise.all(
      users.map(async (user) => {
        const count = await Log.countDocuments({ user: user._id })
        return { ...user.toObject(), totalImages: count }
      })
    )
    res.json({ success: true, data: usersWithStats })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

module.exports = { getHistory, getLogById, getAllLogs, getAllUsers }
