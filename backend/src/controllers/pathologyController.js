const axios = require('axios')
const FormData = require('form-data')

const detectPathology = async (req, res) => {
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
      `${process.env.FASTAPI_URL}/detect-pathology`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 120000
      }
    )

    const result = fastApiResponse.data

    if (result.status === 'error') {
      return res.status(400).json({
        success: false,
        message: result.message
      })
    }

    res.json({
      success: true,
      pathologies: result.pathologies,
      summary: result.summary || null,
      heatmap: result.heatmap,
      bbox: result.bbox || null,
      localization_note: result.localization_note || null,
      disclaimer: result.disclaimer || null,
      warning: result.warning
    })

  } catch (error) {
    const statusCode = error.response?.status || 500
    const message = error.response?.data?.message ||
                    error.response?.data?.detail ||
                    error.message

    res.status(statusCode).json({
      success: false,
      message: `Pathology detection failed: ${message}`
    })
  }
}

module.exports = { detectPathology }
