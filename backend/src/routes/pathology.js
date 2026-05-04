const express = require('express')
const router = express.Router()
const { detectPathology } = require('../controllers/pathologyController')
const { protect } = require('../middleware/auth')
const upload = require('../middleware/upload')

router.post('/', protect, upload.single('file'), detectPathology)

module.exports = router
