const User = require('../models/User')
const Log = require('../models/Log')
const { getPresignedUrl, deleteObject, objectExists, listObjects } = require('../config/minio')

const getStats = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments()
    const totalLogs = await Log.countDocuments()
    const successLogs = await Log.countDocuments({ status: 'success' })
    const failedLogs = await Log.countDocuments({ status: 'failed' })
    const avgTimeResult = await Log.aggregate([
      { $match: { status: 'success' } },
      { $group: { _id: null, avg: { $avg: '$processingTime' } } }
    ])
    const storageByCategory = await Log.aggregate([
      { $match: { status: 'success' } },
      {
        $addFields: {
          category: {
            $switch: {
              branches: [
                { case: { $regexMatch: { input: '$classification', regex: /chest/i }}, then: 'Chest' },
                { case: { $regexMatch: { input: '$classification', regex: /dental/i }}, then: 'Dental' },
                { case: { $regexMatch: { input: '$classification', regex: /pelvic/i }}, then: 'Pelvic' },
                { case: { $regexMatch: { input: '$classification', regex: /skull/i }}, then: 'Skull' }
              ],
              default: 'Other'
            }
          }
        }
      },
      { $group: { _id: '$category', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ])
    const recentActivity = await Log.find()
      .populate('user', 'name email')
      .sort({ createdAt: -1 })
      .limit(10)
    res.json({
      success: true,
      data: {
        users: { total: totalUsers },
        images: {
          total: totalLogs,
          success: successLogs,
          failed: failedLogs,
          avgProcessingTime: avgTimeResult[0]?.avg || 0
        },
        storageByCategory,
        recentActivity
      }
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getUsers = async (req, res) => {
  try {
    const users = await User.find()
      .select('-password')
      .sort({ createdAt: -1 })
    const usersWithStats = await Promise.all(
      users.map(async (user) => {
        const total = await Log.countDocuments({ user: user._id })
        const success = await Log.countDocuments({ user: user._id, status: 'success' })
        return { 
          ...user.toObject(), 
          totalImages: total, 
          successImages: success 
        }
      })
    )
    res.json({ success: true, data: usersWithStats })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const deleteUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' })
    }
    if (user.role === 'responsable') {
      return res.status(400).json({
        success: false,
        message: 'Cannot delete responsable accounts'
      })
    }

    const userLogs = await Log.find({ user: user._id })
    let deletedImagesCount = 0

    for (const log of userLogs) {
      if (log.minioKey) {
        try {
          await deleteObject(log.minioKey)
          deletedImagesCount++
        } catch (err) {
          console.error(`Failed to delete MinIO object ${log.minioKey}:`, err)
        }
      }
    }

    await Log.deleteMany({ user: user._id })
    await User.findByIdAndDelete(req.params.id)

    res.json({
      success: true,
      message: `User deleted successfully`,
      deleted_images_count: deletedImagesCount
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getUserImages = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' })
    }

    const logs = await Log.find({ user: user._id, status: 'success' })
      .sort({ createdAt: -1 })

    const imagesWithUrls = await Promise.all(
      logs.map(async (log) => {
        let presignedUrl = null
        if (log.minioKey) {
          try {
            presignedUrl = await getPresignedUrl(log.minioKey, 3600)
          } catch (err) {
            console.error(`Failed to get presigned URL for ${log.minioKey}:`, err)
          }
        }
        return {
          image_id: log._id,
          filename: log.originalFilename,
          upload_date: log.createdAt,
          presigned_url: presignedUrl,
          classification: log.classification,
          redacted: log.redacted,
          minio_key: log.minioKey
        }
      })
    )

    res.json({
      success: true,
      user: { id: user._id, name: user.name, email: user.email },
      data: imagesWithUrls
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getImageDownload = async (req, res) => {
  try {
    console.log('[DEBUG] Download requested for imageId:', req.params.imageId)
    console.log('[DEBUG] User:', req.user?.email, 'Role:', req.user?.role)
    
    const log = await Log.findById(req.params.imageId)
    console.log('[DEBUG] Log found:', log ? 'YES' : 'NO')
    
    if (!log) {
      return res.status(404).json({ success: false, message: 'Image not found' })
    }

    console.log('[DEBUG] minioKey:', log.minioKey)
    console.log('[DEBUG] minioUri:', log.minioUri)
    console.log('[DEBUG] anonymizedFilename:', log.anonymizedFilename)
    
    let minioKey = log.minioKey
    
    // If minioKey is missing, try to extract from minioUri
    if (!minioKey && log.minioUri) {
      const bucketName = process.env.MINIO_BUCKET || 'anonymized-images'
      const parts = log.minioUri.split(`/${bucketName}/`)
      if (parts.length > 1) {
        minioKey = parts[1]
        console.log('[DEBUG] Extracted minioKey from URI:', minioKey)
        // Save it for future use
        await Log.findByIdAndUpdate(log._id, { minioKey })
      }
    }
    
    // If still no minioKey, try to find by filename in MinIO
    if (!minioKey && log.anonymizedFilename) {
      console.log('[DEBUG] Searching MinIO for filename:', log.anonymizedFilename)
      const allObjects = await listObjects()
      const matches = allObjects.filter(obj => obj.endsWith(log.anonymizedFilename))
      
      if (matches.length > 0) {
        // Prefer match with classification folder
        const classification = (log.classification || '').toLowerCase()
        const classMatch = matches.find(m => m.toLowerCase().includes(classification))
        minioKey = classMatch || matches[0]
        console.log('[DEBUG] Found minioKey by filename search:', minioKey)
        // Save it for future use
        await Log.findByIdAndUpdate(log._id, { minioKey })
      }
    }
    
    if (!minioKey) {
      return res.status(404).json({ success: false, message: 'Image file not found in storage - no minioKey' })
    }

    // Verify object exists in MinIO
    const exists = await objectExists(minioKey)
    console.log('[DEBUG] Object exists in MinIO:', exists)
    
    if (!exists) {
      return res.status(404).json({ 
        success: false, 
        message: `Image file not found in storage: ${minioKey}` 
      })
    }

    const presignedUrl = await getPresignedUrl(minioKey, 3600)
    console.log('[DEBUG] Presigned URL generated:', presignedUrl ? 'YES' : 'NO')

    res.json({
      success: true,
      presigned_url: presignedUrl,
      filename: log.originalFilename,
      classification: log.classification,
      redacted: log.redacted,
      upload_date: log.createdAt
    })
  } catch (error) {
    console.error('[DEBUG] Download error:', error.message)
    res.status(500).json({ success: false, message: error.message })
  }
}

const deleteImage = async (req, res) => {
  try {
    const log = await Log.findById(req.params.imageId)
    if (!log) {
      return res.status(404).json({ success: false, message: 'Image not found' })
    }

    if (log.minioKey) {
      try {
        await deleteObject(log.minioKey)
      } catch (err) {
        console.error(`Failed to delete MinIO object ${log.minioKey}:`, err)
        return res.status(500).json({
          success: false,
          message: 'Failed to delete image from storage'
        })
      }
    }

    await Log.findByIdAndDelete(req.params.imageId)

    res.json({
      success: true,
      message: 'Image deleted successfully'
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const changeRole = async (req, res) => {
  try {
    const { role } = req.body
    if (!['utilisateur', 'utilisateur_medical', 'responsable'].includes(role)) {
      return res.status(400).json({ success: false, message: 'Invalid role' })
    }
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { role },
      { new: true }
    ).select('-password')
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' })
    }
    res.json({ success: true, data: user })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getLogs = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 15
    const skip = (page - 1) * limit
    const { status, search } = req.query

    const filter = {}
    if (status && status !== 'all') filter.status = status
    if (search) {
      filter.originalFilename = { $regex: search, $options: 'i' }
    }

    const logs = await Log.find(filter)
      .populate('user', 'name email')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)

    const total = await Log.countDocuments(filter)

    res.json({
      success: true,
      data: logs,
      pagination: {
        page, limit, total,
        pages: Math.ceil(total / limit)
      }
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const getSettings = async (req, res) => {
  try {
    const settings = {
      ocrConfidenceThreshold: 0.1,
      borderMarginPx: 100,
      inpaintRadius: 1,
      iouThreshold: 0.5,
      paddleOcrEnabled: true,
      easyOcrEnabled: true,
      dicomTagsToRemove: [
        'PatientName', 'PatientID', 'PatientBirthDate',
        'PatientSex', 'PatientAge', 'InstitutionName',
        'ReferringPhysicianName', 'StudyDate',
        'SeriesDate', 'AcquisitionDate',
        'ContentDate', 'AccessionNumber'
      ]
    }
    res.json({ success: true, data: settings })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

const updateSettings = async (req, res) => {
  try {
    res.json({
      success: true,
      message: 'Settings updated successfully',
      data: req.body
    })
  } catch (error) {
    res.status(500).json({ success: false, message: error.message })
  }
}

module.exports = {
  getStats, getUsers, deleteUser, changeRole,
  getLogs, getSettings, updateSettings,
  getUserImages, deleteImage, getImageDownload
}
