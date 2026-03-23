const express = require('express')
const router = express.Router()
const {
  getHistory,
  getLogById
} = require('../controllers/historyController')
const adminController = require('../controllers/adminController')
const imagesController = require('../controllers/imagesController')
const { protect, responsableOnly, medicalOrResponsable } = require('../middleware/auth')

router.get('/', protect, getHistory)

router.get('/images/all', protect, medicalOrResponsable, imagesController.getAllImages)
router.get('/images/:imageId/download', protect, medicalOrResponsable, adminController.getImageDownload)

router.get('/admin/stats', protect, responsableOnly, adminController.getStats)
router.get('/admin/users', protect, responsableOnly, adminController.getUsers)
router.get('/admin/logs', protect, responsableOnly, adminController.getLogs)
router.get('/admin/settings', protect, responsableOnly, adminController.getSettings)
router.put('/admin/settings', protect, responsableOnly, adminController.updateSettings)
router.delete('/admin/users/:id', protect, responsableOnly, adminController.deleteUser)
router.put('/admin/users/:id/role', protect, responsableOnly, adminController.changeRole)
router.get('/admin/users/:id/images', protect, responsableOnly, adminController.getUserImages)
router.get('/admin/images/:imageId/download', protect, responsableOnly, adminController.getImageDownload)
router.delete('/admin/images/:imageId', protect, responsableOnly, adminController.deleteImage)

router.get('/:id', protect, getLogById)

module.exports = router
