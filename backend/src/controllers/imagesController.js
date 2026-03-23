const Log = require('../models/Log')
const { getPresignedUrl } = require('../config/minio')

const getAllImages = async (req, res) => {
  try {
    const logs = await Log.find({ status: 'success' })
      .populate('user', 'name email')
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
          uploaded_by: log.user?.name || 'Unknown',
          upload_date: log.createdAt,
          presigned_url: presignedUrl,
          classification: log.classification,
          redacted: log.redacted
        }
      })
    )

    res.json({
      success: true,
      data: imagesWithUrls
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    })
  }
}

module.exports = { getAllImages }
