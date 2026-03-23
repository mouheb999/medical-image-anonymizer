const express = require('express')
const router = express.Router()
const { anonymizeImage } = require('../controllers/anonymizeController')
const { protect } = require('../middleware/auth')
const upload = require('../middleware/upload')

router.post('/', protect, upload.single('file'), anonymizeImage)

module.exports = router
