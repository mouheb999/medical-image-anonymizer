const mongoose = require('mongoose')

const LogSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  originalFilename: {
    type: String,
    required: true
  },
  anonymizedFilename: {
    type: String
  },
  classification: {
    type: String
  },
  confidence: {
    type: Number
  },
  format: {
    type: String
  },
  isDicom: {
    type: Boolean,
    default: false
  },
  tagsAnonymized: {
    type: Number,
    default: 0
  },
  paddleRegions: {
    type: Number,
    default: 0
  },
  easyRegions: {
    type: Number,
    default: 0
  },
  totalRegions: {
    type: Number,
    default: 0
  },
  redacted: {
    type: Number,
    default: 0
  },
  skipped: {
    type: Number,
    default: 0
  },
  minioUri: {
    type: String
  },
  minioKey: {
    type: String
  },
  downloadUrl: {
    type: String
  },
  processingTime: {
    type: Number
  },
  status: {
    type: String,
    enum: ['success', 'failed', 'pending'],
    default: 'pending'
  },
  errorMessage: {
    type: String
  }
}, {
  timestamps: true
})

module.exports = mongoose.model('Log', LogSchema)
