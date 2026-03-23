/**
 * Migration script to fix existing DB records with wrong minioKey
 * 
 * The issue: minioKey was not being saved, or was saved without the category folder prefix
 * The fix: Extract the correct key from minioUri or match against actual MinIO objects
 * 
 * Run with: node scripts/fix-minio-keys.js
 */

require('dotenv').config()
const mongoose = require('mongoose')
const { Client } = require('minio')

// MinIO client
const minioClient = new Client({
  endPoint: process.env.MINIO_ENDPOINT?.split(':')[0] || 'localhost',
  port: parseInt(process.env.MINIO_ENDPOINT?.split(':')[1]) || 9000,
  useSSL: process.env.MINIO_USE_SSL === 'true',
  accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
})

const BUCKET = process.env.MINIO_BUCKET || 'anonymized-images'

// Log schema (simplified for this script)
const LogSchema = new mongoose.Schema({
  originalFilename: String,
  anonymizedFilename: String,
  classification: String,
  minioUri: String,
  minioKey: String,
  status: String
}, { timestamps: true })

const Log = mongoose.model('Log', LogSchema)

async function listMinioObjects() {
  return new Promise((resolve, reject) => {
    const objects = []
    const stream = minioClient.listObjects(BUCKET, '', true)
    
    stream.on('data', (obj) => {
      objects.push(obj.name)
    })
    stream.on('error', reject)
    stream.on('end', () => resolve(objects))
  })
}

async function main() {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/medical_anonymizer')
    console.log('Connected to MongoDB')

    // List all objects in MinIO
    console.log(`\nListing objects in MinIO bucket: ${BUCKET}`)
    const minioObjects = await listMinioObjects()
    console.log(`Found ${minioObjects.length} objects in MinIO:`)
    minioObjects.forEach(obj => console.log(`  ${obj}`))

    // Get all logs from DB
    const logs = await Log.find({})
    console.log(`\nFound ${logs.length} logs in database`)

    let fixed = 0
    let alreadyCorrect = 0
    let noMatch = 0

    for (const log of logs) {
      console.log(`\n--- Log ID: ${log._id} ---`)
      console.log(`  originalFilename: ${log.originalFilename}`)
      console.log(`  anonymizedFilename: ${log.anonymizedFilename}`)
      console.log(`  classification: ${log.classification}`)
      console.log(`  minioUri: ${log.minioUri}`)
      console.log(`  minioKey: ${log.minioKey}`)

      // Skip if no minioUri (upload failed)
      if (!log.minioUri) {
        console.log(`  ⚠ No minioUri - skipping (upload may have failed)`)
        continue
      }

      // Try to extract minioKey from minioUri
      let extractedKey = null
      const parts = log.minioUri.split(`/${BUCKET}/`)
      if (parts.length > 1) {
        extractedKey = parts[1]
      }
      console.log(`  Extracted key from URI: ${extractedKey}`)

      // Check if minioKey is already correct
      if (log.minioKey && minioObjects.includes(log.minioKey)) {
        console.log(`  ✓ minioKey already correct and exists in MinIO`)
        alreadyCorrect++
        continue
      }

      // Check if extracted key exists in MinIO
      if (extractedKey && minioObjects.includes(extractedKey)) {
        console.log(`  → Fixing: setting minioKey to '${extractedKey}'`)
        await Log.findByIdAndUpdate(log._id, { minioKey: extractedKey })
        fixed++
        continue
      }

      // Try to find by filename match
      const filename = log.anonymizedFilename || log.originalFilename
      if (filename) {
        const matches = minioObjects.filter(obj => obj.endsWith(filename))
        
        if (matches.length === 1) {
          console.log(`  → Fixing by filename match: '${matches[0]}'`)
          await Log.findByIdAndUpdate(log._id, { minioKey: matches[0] })
          fixed++
          continue
        } else if (matches.length > 1) {
          // Use classification to pick the right one
          const classification = (log.classification || '').toLowerCase()
          const classMatch = matches.find(m => m.toLowerCase().includes(classification))
          
          if (classMatch) {
            console.log(`  → Fixing by classification match: '${classMatch}'`)
            await Log.findByIdAndUpdate(log._id, { minioKey: classMatch })
            fixed++
            continue
          }
          
          console.log(`  ⚠ Multiple matches found: ${matches.join(', ')} - using first`)
          await Log.findByIdAndUpdate(log._id, { minioKey: matches[0] })
          fixed++
          continue
        }
      }

      console.log(`  ✗ No match found in MinIO`)
      noMatch++
    }

    console.log(`\n========== Summary ==========`)
    console.log(`Total logs: ${logs.length}`)
    console.log(`Already correct: ${alreadyCorrect}`)
    console.log(`Fixed: ${fixed}`)
    console.log(`No match found: ${noMatch}`)
    console.log(`==============================`)

  } catch (error) {
    console.error('Error:', error)
  } finally {
    await mongoose.disconnect()
    console.log('\nDisconnected from MongoDB')
  }
}

main()
