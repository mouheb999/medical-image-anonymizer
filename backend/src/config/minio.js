const Minio = require('minio')

const minioClient = new Minio.Client({
  endPoint: process.env.MINIO_ENDPOINT?.split(':')[0] || 'localhost',
  port: parseInt(process.env.MINIO_ENDPOINT?.split(':')[1]) || 9000,
  useSSL: process.env.MINIO_USE_SSL === 'true',
  accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
})

const BUCKET_NAME = process.env.MINIO_BUCKET || 'anonymized-images'

const getPresignedUrl = async (objectName, expirySeconds = 3600) => {
  try {
    return await minioClient.presignedGetObject(BUCKET_NAME, objectName, expirySeconds)
  } catch (error) {
    console.error('Error generating presigned URL:', error)
    throw error
  }
}

const deleteObject = async (objectName) => {
  try {
    await minioClient.removeObject(BUCKET_NAME, objectName)
    return true
  } catch (error) {
    console.error('Error deleting object from MinIO:', error)
    throw error
  }
}

const objectExists = async (objectName) => {
  try {
    await minioClient.statObject(BUCKET_NAME, objectName)
    return true
  } catch (error) {
    if (error.code === 'NotFound') {
      return false
    }
    throw error
  }
}

const listObjects = async (prefix = '') => {
  return new Promise((resolve, reject) => {
    const objects = []
    const stream = minioClient.listObjects(BUCKET_NAME, prefix, true)
    stream.on('data', (obj) => objects.push(obj.name))
    stream.on('error', reject)
    stream.on('end', () => resolve(objects))
  })
}

const ensureBucket = async () => {
  try {
    const exists = await minioClient.bucketExists(BUCKET_NAME)
    if (!exists) {
      await minioClient.makeBucket(BUCKET_NAME)
      console.log(`Bucket ${BUCKET_NAME} created`)
    }
  } catch (error) {
    console.error('Error ensuring bucket exists:', error)
  }
}

module.exports = {
  minioClient,
  BUCKET_NAME,
  getPresignedUrl,
  deleteObject,
  objectExists,
  listObjects,
  ensureBucket
}
