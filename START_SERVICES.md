# Quick Start Guide - Medical Image Anonymizer

Use these commands in **4 separate terminals** to run the full stack.

---

## Terminal 1: MongoDB
**Start MongoDB service** (if not already running as a service)

```powershell
# Windows - if MongoDB is installed as a service, it should auto-start
# Otherwise, start it manually:
mongod
```

**Check MongoDB Status:**
```powershell
mongosh
# Then in mongo shell:
show dbs
use medical_anonymizer
show collections
db.users.countDocuments()
db.logs.countDocuments()
exit
```

---

## Terminal 2: Node.js Backend
**Port:** 5000

```powershell
cd c:\Users\MSI\Desktop\PFE_Test\backend
npm run dev
```

**Expected Output:**
```
MongoDB Connected: localhost
╔════════════════════════════════════════╗
║   Medical Anonymizer Backend           ║
║   Server running on port 5000          ║
╚════════════════════════════════════════╝
```

---

## Terminal 3: React Frontend
**Port:** 3000

```powershell
cd c:\Users\MSI\Desktop\PFE_Test\client
npm run dev
```

**Expected Output:**
```
VITE v5.4.21  ready in 524 ms
➜  Local:   http://localhost:3000/
```

---

## Terminal 4: FastAPI AI Pipeline
**Port:** 8000

```powershell
cd c:\Users\MSI\Desktop\PFE_Test
venv\Scripts\activate
python -m uvicorn api.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

## Terminal 5 (Optional): MinIO Storage
**Port:** 9000 (API), 9001 (Console)

```powershell
cd C:\minio
.\minio.exe server C:\minio\data --console-address ":9001"
```

**Expected Output:**
```
API: http://127.0.0.1:9000
WebUI: http://127.0.0.1:9001
RootUser: minioadmin
RootPass: minioadmin
```

---

## Quick Health Checks

### Check Backend
```powershell
curl http://localhost:5000/api/health
```

### Check FastAPI
```powershell
curl http://localhost:8000/health
```

### Check Frontend
Open browser: http://localhost:3000

### Check MinIO Console
Open browser: http://localhost:9001

---

## Stop All Services

Press **Ctrl+C** in each terminal window.

---

## Troubleshooting

### Backend won't start
- Check MongoDB is running: `mongosh`
- Check port 5000 is free: `netstat -ano | findstr :5000`

### Frontend won't start
- Check port 3000 is free: `netstat -ano | findstr :3000`
- Delete node_modules and reinstall: `npm install`

### FastAPI won't start
- Activate venv first: `venv\Scripts\activate`
- Check Python version: `python --version` (should be 3.10+)
- Check port 8000 is free: `netstat -ano | findstr :8000`

### MinIO won't start
- Check ports 9000/9001 are free
- Verify minio.exe exists in C:\minio

---

## Full Stack Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                         │
│              http://localhost:3000                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              React Frontend (Vite)                      │
│                   Port 3000                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Node.js Backend (Express)                       │
│                   Port 5000                             │
│  - JWT Authentication                                   │
│  - User Management                                      │
│  - History Logging                                      │
└─────────┬──────────────────────────┬────────────────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐      ┌──────────────────────────────┐
│    MongoDB       │      │   FastAPI AI Pipeline        │
│   Port 27017     │      │      Port 8000               │
│  - Users         │      │  - CLIP Classification       │
│  - Logs          │      │  - PaddleOCR + EasyOCR       │
└──────────────────┘      │  - Pixel Redaction           │
                          │  - DICOM Processing          │
                          └──────────┬───────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────────┐
                          │      MinIO Storage           │
                          │   Port 9000 (API)            │
                          │   Port 9001 (Console)        │
                          │  - Anonymized Images         │
                          └──────────────────────────────┘
```

---

## Development Workflow

1. **Start all services** (4-5 terminals)
2. **Open browser** to http://localhost:3000
3. **Register** a new user
4. **Upload** a medical image
5. **View results** with anonymization stats
6. **Check history** for all processed images
7. **Download** anonymized files

All services have **hot-reload** enabled for development!
