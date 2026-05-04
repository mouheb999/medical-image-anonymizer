const axios = require('axios')
const FormData = require('form-data')
const Log = require('../models/Log')

const anonymizeImage = async (req, res) => {
  const startTime = Date.now()
  
  const log = await Log.create({
    user: req.user._id,
    originalFilename: req.file.originalname,
    status: 'pending'
  })
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No file uploaded'
      })
    }
    
    // Extract OCR parameters from request body (sent by frontend)
    const conf_threshold = parseFloat(req.body.conf_threshold) || 0.1
    const padding = parseInt(req.body.padding) || 5
    const border_margin = parseInt(req.body.border_margin) || 100
    const border_pct = parseFloat(req.body.border_pct) || 0.20
    
    console.log('[DEBUG] OCR Parameters from frontend:', {
      conf_threshold,
      padding,
      border_margin,
      border_pct
    })
    
    const formData = new FormData()
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    })
    
    // Forward parameters to FastAPI
    formData.append('conf_threshold', conf_threshold.toString())
    formData.append('padding', padding.toString())
    formData.append('border_margin', border_margin.toString())
    formData.append('border_pct', border_pct.toString())
    
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
    
    // Extract minioKey from minio_uri
    // minio_uri format: minio://localhost:9000/anonymized-images/chest/anonymized_file.jpg
    // minioKey should be: chest/anonymized_file.jpg
    let minioKey = null
    if (result.minio_uri) {
      const bucketName = process.env.MINIO_BUCKET || 'anonymized-images'
      const parts = result.minio_uri.split(`/${bucketName}/`)
      if (parts.length > 1) {
        minioKey = parts[1]
      }
      console.log('[DEBUG] Extracted minioKey:', minioKey, 'from minio_uri:', result.minio_uri)
    }
    
    await Log.findByIdAndUpdate(log._id, {
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
      minioKey: minioKey,
      downloadUrl: result.download_url,
      processingTime,
      status: 'success'
    })
    
    res.json({
      success: true,
      logId: log._id,
      processingTime,
      ...result
    })
    
  } catch (error) {
    await Log.findByIdAndUpdate(log._id, {
      status: 'failed',
      errorMessage: error.response?.data?.detail || error.message
    })
    
    const statusCode = error.response?.status || 500
    const message = error.response?.data?.detail || 
                    error.response?.data?.error ||
                    error.message
    
    res.status(statusCode).json({
      success: false,
      message: `Processing failed: ${message}` 
    })
  }
}

module.exports = { anonymizeImage }
