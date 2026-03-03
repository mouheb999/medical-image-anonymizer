# MinIO S3 Storage Integration Documentation

## Overview

This document describes the MinIO S3 storage integration for the Medical Image Anonymization API. After successfully anonymizing medical images, the system automatically uploads results to MinIO object storage and provides presigned download URLs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Medical Image Upload                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              7-Stage Anonymization Pipeline                     │
│  1. Classification  →  2. Validation  →  3. Metadata Cleaning   │
│  4. Preprocessing   →  5. Dual OCR    →  6. Pixel Redaction     │
│  7. Save Output                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MinIO Upload (Non-Fatal)                       │
│  • Upload to S3 bucket: anonymized-images                       │
│  • Organize by date: YYYY/MM/DD/filename                        │
│  • Generate presigned URL (24h expiry)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Response                                 │
│  • output_filename: Local file path                             │
│  • minio_uri: S3 storage location                               │
│  • download_url: Presigned HTTP URL                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. `api/storage.py` - MinIO Client Wrapper

**Purpose:** Encapsulates MinIO S3 operations for uploading files and generating presigned URLs.

**Key Features:**
- Automatic bucket creation
- Date-organized storage (YYYY/MM/DD)
- Presigned URL generation with configurable expiry
- Error handling with detailed logging

**Class: `MinIOStorage`**

```python
class MinIOStorage:
    def __init__(
        self,
        endpoint: str,           # e.g., "localhost:9000"
        access_key: str,         # MinIO access key
        secret_key: str,         # MinIO secret key
        bucket_name: str,        # Target bucket
        secure: bool = False     # Use HTTPS
    )
    
    def upload_file(
        self,
        file_path: str,          # Local file to upload
        object_name: str = None  # Optional custom name
    ) -> str:                    # Returns: minio://endpoint/bucket/path
    
    def get_url(
        self,
        object_name: str,        # Object path in bucket
        expires_hours: int = 24  # URL expiry time
    ) -> str:                    # Returns: presigned HTTP URL
```

**Example Usage:**

```python
from api.storage import MinIOStorage

storage = MinIOStorage(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    bucket_name="anonymized-images"
)

# Upload file
uri = storage.upload_file("/path/to/anonymized_image.jpg")
# Returns: "minio://localhost:9000/anonymized-images/2026/03/02/anonymized_image.jpg"

# Get presigned download URL
object_name = "2026/03/02/anonymized_image.jpg"
url = storage.get_url(object_name, expires_hours=24)
# Returns: "http://localhost:9000/anonymized-images/2026/03/02/anonymized_image.jpg?X-Amz-..."
```

---

### 2. `api/config.py` - Environment Configuration

**Purpose:** Centralized configuration management using environment variables.

**Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `anonymized-images` | Target bucket name |
| `MINIO_SECURE` | `false` | Use HTTPS (true/false) |
| `OUTPUT_DIR` | `./output` | Local output directory |
| `TEMP_DIR` | `./api/temp` | Temporary upload directory |

**Usage:**

```python
from api.config import settings

print(settings.minio_endpoint)  # "localhost:9000"
print(settings.minio_bucket)    # "anonymized-images"
```

---

### 3. `.env` - Environment Variables File

**Location:** Project root (`c:\Users\MSI\Desktop\PFE_Test\.env`)

**Contents:**

```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=anonymized-images
MINIO_SECURE=false
OUTPUT_DIR=./output
TEMP_DIR=./api/temp
```

**Security Notes:**
- ⚠️ **Never commit `.env` to Git** - Add to `.gitignore`
- 🔒 Change default credentials in production
- 🔐 Use strong passwords for `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`

---

### 4. `api/main.py` - Integration Point

**Modified Sections:**

#### A. Environment Loading (Top of file)

```python
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
```

#### B. MinIO Upload Logic (After Stage 7: Save Output)

```python
# Upload to MinIO
minio_uri = None
download_url = None
try:
    from api.storage import MinIOStorage
    from api.config import settings
    
    storage = MinIOStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket,
        secure=settings.minio_secure
    )
    
    minio_uri = storage.upload_file(str(output_path))
    
    # Extract object name from URI for presigned URL
    object_name = minio_uri.split(f"/{settings.minio_bucket}/", 1)[1]
    download_url = storage.get_url(object_name, expires_hours=24)
    
    logger.info(f"Uploaded to MinIO: {minio_uri}")
except Exception as e:
    logger.warning(f"MinIO upload failed (non-fatal): {e}")
    minio_uri = None
    download_url = None
```

#### C. Enhanced API Response

```python
response_data = {
    "status": "success",
    "classification": category,
    "confidence": float(confidence),
    "format": image_format,
    "tags_anonymized": tags_anonymized,
    "paddle_regions": paddle_count,
    "easy_regions": easy_count,
    "total_regions": merged_count,
    "redacted": redacted_count,
    "skipped": skipped_count,
    "output_filename": output_filename
}

# Add MinIO fields if upload succeeded
if minio_uri:
    response_data["minio_uri"] = minio_uri
    response_data["download_url"] = download_url

return response_data
```

---

## API Response Format

### Success Response (With MinIO)

```json
{
  "status": "success",
  "classification": "Accepted: a chest x-ray radiograph showing lungs and ribs",
  "confidence": 0.93,
  "format": "JPEG",
  "tags_anonymized": 0,
  "paddle_regions": 6,
  "easy_regions": 5,
  "total_regions": 6,
  "redacted": 6,
  "skipped": 0,
  "output_filename": "anonymized_person49_virus_101.jpeg",
  "minio_uri": "minio://localhost:9000/anonymized-images/2026/03/02/anonymized_person49_virus_101.jpeg",
  "download_url": "http://localhost:9000/anonymized-images/2026/03/02/anonymized_person49_virus_101.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20260302%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260302T001234Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=..."
}
```

### Success Response (MinIO Failed - Non-Fatal)

```json
{
  "status": "success",
  "classification": "Accepted: a chest x-ray radiograph showing lungs and ribs",
  "confidence": 0.93,
  "format": "JPEG",
  "tags_anonymized": 0,
  "paddle_regions": 6,
  "easy_regions": 5,
  "total_regions": 6,
  "redacted": 6,
  "skipped": 0,
  "output_filename": "anonymized_person49_virus_101.jpeg"
}
```

**Note:** If MinIO upload fails, the API still returns successfully with the anonymized image saved locally. The `minio_uri` and `download_url` fields are simply omitted.

---

## Storage Organization

### Bucket Structure

```
anonymized-images/
├── 2026/
│   ├── 02/
│   │   ├── 25/
│   │   │   ├── anonymized_person49_virus_101.jpeg
│   │   │   ├── anonymized_person1656_virus_2862.jpeg
│   │   │   └── anonymized_test_image.png
│   │   └── 28/
│   │       └── anonymized_chest_xray.dcm
│   └── 03/
│       ├── 01/
│       │   └── anonymized_sample.jpg
│       └── 02/
│           └── anonymized_medical_scan.png
```

**Benefits:**
- ✅ Chronological organization
- ✅ Easy to find recent uploads
- ✅ Supports archival policies by date
- ✅ Prevents filename collisions

---

## Deployment

### Option 1: Docker Compose (Recommended)

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  minio:
    image: minio/minio
    container_name: minio
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  minio_data:
```

**Start MinIO:**

```bash
docker compose up -d minio
```

**Stop MinIO:**

```bash
docker compose down minio
```

---

### Option 2: Standalone MinIO (Windows)

**Download and Run:**

```powershell
# Create directory
mkdir C:\minio
cd C:\minio

# Download MinIO
Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "minio.exe"

# Create data directory
mkdir C:\minio\data

# Set credentials
$env:MINIO_ROOT_USER="minioadmin"
$env:MINIO_ROOT_PASSWORD="minioadmin"

# Start server
.\minio.exe server C:\minio\data --console-address ":9001"
```

**Keep the terminal window open** - MinIO runs in the foreground.

---

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **MinIO API** | http://localhost:9000 | Access via SDK |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **FastAPI** | http://localhost:8000 | N/A |
| **API Docs** | http://localhost:8000/docs | N/A |

---

## Testing

### 1. Test MinIO Health

```powershell
Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live"
```

**Expected:** `StatusCode: 200 OK`

---

### 2. Test API with MinIO

**Upload via cURL:**

```bash
curl -X POST "http://localhost:8000/anonymize" \
  -F "file=@person49_virus_101.jpeg"
```

**Expected Response:**

```json
{
  "status": "success",
  "minio_uri": "minio://localhost:9000/anonymized-images/2026/03/02/anonymized_person49_virus_101.jpeg",
  "download_url": "http://localhost:9000/anonymized-images/..."
}
```

---

### 3. Verify in MinIO Console

1. Open http://localhost:9001
2. Login: `minioadmin` / `minioadmin`
3. Navigate to **Buckets** → `anonymized-images`
4. Check for uploaded file in `YYYY/MM/DD/` folder

---

### 4. Test Presigned URL

Copy the `download_url` from the API response and paste it in your browser. The anonymized image should download directly without authentication.

---

## Error Handling

### Non-Fatal Failure Design

The MinIO upload is wrapped in a try-except block to ensure the API remains functional even if MinIO is unavailable:

```python
try:
    minio_uri = storage.upload_file(str(output_path))
except Exception as e:
    logger.warning(f"MinIO upload failed (non-fatal): {e}")
    minio_uri = None
```

**Behavior:**
- ✅ API returns 200 OK with anonymized image
- ⚠️ `minio_uri` and `download_url` fields are omitted
- 📝 Warning logged to server logs

---

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Connection refused` | MinIO not running | Start MinIO server |
| `Access Denied` | Wrong credentials | Check `.env` file |
| `Bucket does not exist` | Bucket not created | MinIOStorage auto-creates it |
| `ImportError: minio` | Package not installed | `pip install minio` |

---

## Security Considerations

### Production Deployment

1. **Change Default Credentials:**
   ```env
   MINIO_ROOT_USER=your_secure_username
   MINIO_ROOT_PASSWORD=your_strong_password_here
   ```

2. **Enable HTTPS:**
   ```env
   MINIO_SECURE=true
   MINIO_ENDPOINT=minio.yourdomain.com:9000
   ```

3. **Use TLS Certificates:**
   - Configure MinIO with valid SSL certificates
   - Update `secure=True` in `MinIOStorage` initialization

4. **Restrict Access:**
   - Use MinIO IAM policies
   - Create service accounts with limited permissions
   - Enable bucket versioning for audit trails

5. **Network Security:**
   - Use firewall rules to restrict MinIO access
   - Deploy MinIO in private network
   - Use VPN or bastion host for console access

---

## Performance Optimization

### 1. Parallel Uploads (Future Enhancement)

For high-throughput scenarios, consider async uploads:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def upload_async(file_path):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, storage.upload_file, file_path
        )
```

### 2. Multipart Upload (Large Files)

MinIO automatically handles multipart uploads for files >5MB. No code changes needed.

### 3. Presigned URL Caching

Cache presigned URLs in Redis to avoid regenerating them:

```python
import redis
cache = redis.Redis(host='localhost', port=6379)

def get_cached_url(object_name):
    cached = cache.get(f"url:{object_name}")
    if cached:
        return cached.decode()
    
    url = storage.get_url(object_name)
    cache.setex(f"url:{object_name}", 86400, url)  # 24h expiry
    return url
```

---

## Monitoring

### MinIO Metrics

MinIO exposes Prometheus metrics at:
```
http://localhost:9000/minio/v2/metrics/cluster
```

**Key Metrics:**
- `minio_bucket_usage_total_bytes` - Storage usage
- `minio_s3_requests_total` - Request count
- `minio_s3_errors_total` - Error count

### API Logging

Check logs for MinIO operations:

```bash
# Successful upload
INFO - Uploaded to MinIO: minio://localhost:9000/anonymized-images/2026/03/02/file.jpg

# Failed upload (non-fatal)
WARNING - MinIO upload failed (non-fatal): Connection refused
```

---

## Troubleshooting

### MinIO Not Starting

**Symptom:** `Connection refused` errors

**Check:**
```powershell
Test-NetConnection -ComputerName localhost -Port 9000
```

**Solution:**
- Ensure MinIO process is running
- Check firewall rules
- Verify port 9000 is not in use

---

### Bucket Not Created

**Symptom:** `Bucket does not exist` error

**Solution:**
The `MinIOStorage` class automatically creates buckets. If it fails:

```python
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

if not client.bucket_exists("anonymized-images"):
    client.make_bucket("anonymized-images")
```

---

### Presigned URL Expired

**Symptom:** 403 Forbidden when accessing download URL

**Cause:** URL expired (default 24 hours)

**Solution:** Generate a new presigned URL:

```python
new_url = storage.get_url(object_name, expires_hours=24)
```

---

## Dependencies

### Python Packages

```txt
minio>=7.1.0
python-dotenv>=1.0.0
```

**Install:**

```bash
pip install minio python-dotenv
```

---

## File Checklist

| File | Purpose | Status |
|------|---------|--------|
| `api/storage.py` | MinIO client wrapper | ✅ Created |
| `api/config.py` | Environment configuration | ✅ Created |
| `api/main.py` | MinIO integration in API | ✅ Modified |
| `.env` | Environment variables | ✅ Created |
| `docker-compose.yml` | MinIO service definition | ✅ Modified |
| `requirements.txt` | Python dependencies | ✅ Updated |

---

## Summary

The MinIO integration provides:

✅ **Automatic S3 storage** for all anonymized images  
✅ **Date-organized structure** for easy management  
✅ **Presigned URLs** for secure, temporary access  
✅ **Non-fatal design** - API works even if MinIO is down  
✅ **Environment-based config** - no hardcoded credentials  
✅ **Production-ready** - supports HTTPS, IAM, and monitoring  

---

## Next Steps

1. ✅ MinIO running on localhost:9000
2. ✅ API integrated with MinIO upload
3. ✅ Frontend displays download URLs
4. 🔄 **Test with real medical images**
5. 🔄 **Configure production credentials**
6. 🔄 **Set up monitoring and alerts**
7. 🔄 **Implement backup and retention policies**

---

**For questions or issues, refer to:**
- MinIO Documentation: https://docs.min.io
- MinIO Python SDK: https://min.io/docs/minio/linux/developers/python/minio-py.html
- Project README: `README.md`
