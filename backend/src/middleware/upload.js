const multer = require('multer')
const path = require('path')

const storage = multer.memoryStorage()

const fileFilter = (req, file, cb) => {
  const allowedTypes = [
    'image/jpeg',
    'image/png',
    'image/bmp',
    'image/tiff',
    'application/octet-stream'
  ]
  const allowedExtensions = [
    '.jpg', '.jpeg', '.png', 
    '.bmp', '.tiff', '.tif', 
    '.dcm', '.dicom'
  ]
  
  const ext = path.extname(file.originalname).toLowerCase()
  
  if (allowedExtensions.includes(ext)) {
    cb(null, true)
  } else {
    cb(new Error(`Unsupported file type: ${ext}`), false)
  }
}

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 50 * 1024 * 1024 }
})

module.exports = upload
