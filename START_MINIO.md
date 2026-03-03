# Running MinIO Without Docker

## Download MinIO for Windows

1. **Download MinIO Server:**
   ```powershell
   # Create a directory for MinIO
   mkdir C:\minio
   cd C:\minio
   
   # Download MinIO (Windows 64-bit)
   Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "minio.exe"
   ```

2. **Create data directory:**
   ```powershell
   mkdir C:\minio\data
   ```

3. **Start MinIO Server:**
   ```powershell
   # Set environment variables
   $env:MINIO_ROOT_USER="minioadmin"
   $env:MINIO_ROOT_PASSWORD="minioadmin"
   
   # Start server
   .\minio.exe server C:\minio\data --console-address ":9001"
   ```

4. **Access MinIO:**
   - **API:** http://localhost:9000
   - **Console:** http://localhost:9001
   - **Login:** minioadmin / minioadmin

## Keep MinIO Running

Open a **new PowerShell window** and leave it running. MinIO will be available at localhost:9000.

## Test the API

In another terminal:
```powershell
cd c:\Users\MSI\Desktop\PFE_Test
venv\Scripts\activate
uvicorn api.main:app --reload --port 8000
```

Then open `frontend/index.html` and upload an image.

---

## Alternative: Update .env to Skip MinIO

If you don't need MinIO right now, the API will still work - it just won't upload to S3.

The MinIO upload is **non-fatal** - if it fails, the API returns results without `minio_uri`.
