# MERN Stack Setup for Medical Image Anonymizer

This document explains how to set up and run the complete MERN stack application on top of the existing FastAPI pipeline.

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   React     │─────▶│   Node.js   │─────▶│   FastAPI   │
│  (Port 3000)│      │  (Port 5000)│      │  (Port 8000)│
└─────────────┘      └─────────────┘      └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   MongoDB   │
                     │ (Port 27017)│
                     └─────────────┘
```

## Prerequisites

- Node.js 18+ installed
- MongoDB installed and running (or use Docker)
- FastAPI pipeline already running on port 8000
- MinIO running on port 9000

## Quick Start (Development)

### 1. Install Backend Dependencies

```bash
cd backend
npm install
```

### 2. Configure Backend Environment

Edit `backend/.env`:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/medical_anonymizer
JWT_SECRET=your_super_secret_jwt_key_change_this_in_production
JWT_EXPIRE=7d
FASTAPI_URL=http://localhost:8000
NODE_ENV=development
```

### 3. Start Backend Server

```bash
cd backend
npm run dev
```

Backend will run on http://localhost:5000

### 4. Install Client Dependencies

```bash
cd client
npm install
```

### 5. Start React Client

```bash
cd client
npm run dev
```

Client will run on http://localhost:3000

### 6. Create First User

1. Open http://localhost:3000
2. Click "Register here"
3. Fill in name, email, password
4. You'll be redirected to the dashboard

## Docker Deployment

### Start All Services

```bash
docker-compose up -d
```

This will start:
- MongoDB (port 27017)
- FastAPI anonymizer (port 8000)
- Node.js backend (port 5000)
- React client (port 3000)
- MinIO (ports 9000, 9001)

### Stop All Services

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f backend
docker-compose logs -f client
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user

### Anonymization
- `POST /api/anonymize` - Upload and anonymize image (requires JWT)

### History
- `GET /api/history` - Get user's anonymization history
- `GET /api/history/:id` - Get specific log details
- `GET /api/history/admin/logs` - Get all logs (admin only)
- `GET /api/history/admin/users` - Get all users (admin only)

## Frontend Routes

- `/login` - Login page
- `/register` - Registration page
- `/dashboard` - Main dashboard with upload zone
- `/history` - Anonymization history
- `/result/:id` - View specific result details

## Database Schema

### User Model
```javascript
{
  name: String,
  email: String (unique),
  password: String (hashed),
  role: String (user/admin),
  isActive: Boolean,
  timestamps: true
}
```

### Log Model
```javascript
{
  user: ObjectId (ref: User),
  originalFilename: String,
  anonymizedFilename: String,
  classification: String,
  confidence: Number,
  format: String,
  isDicom: Boolean,
  tagsAnonymized: Number,
  paddleRegions: Number,
  easyRegions: Number,
  totalRegions: Number,
  redacted: Number,
  skipped: Number,
  minioUri: String,
  downloadUrl: String,
  processingTime: Number,
  status: String (success/failed/pending),
  errorMessage: String,
  timestamps: true
}
```

## Security Features

- JWT authentication with 7-day expiration
- Password hashing with bcrypt (12 rounds)
- Protected routes (requires valid JWT)
- Admin-only routes (requires role='admin')
- CORS enabled for localhost:3000
- File upload validation (type, size)

## Creating an Admin User

MongoDB shell:
```javascript
use medical_anonymizer
db.users.updateOne(
  { email: "your@email.com" },
  { $set: { role: "admin" } }
)
```

## Troubleshooting

### Backend won't start
- Check MongoDB is running: `mongosh`
- Check port 5000 is available
- Verify .env file exists with correct values

### Client won't connect to backend
- Verify backend is running on port 5000
- Check CORS settings in `backend/src/app.js`
- Verify API baseURL in `client/src/api/axios.js`

### FastAPI connection fails
- Ensure FastAPI is running on port 8000
- Check FASTAPI_URL in backend .env
- Verify network connectivity

### File upload fails
- Check file size < 50MB
- Verify file type is supported
- Check FastAPI logs for errors

## Development Tips

### Hot Reload
- Backend: Uses nodemon for auto-restart
- Client: Vite provides instant HMR

### Debugging
- Backend logs: Check terminal running `npm run dev`
- Client logs: Open browser DevTools console
- MongoDB: Use MongoDB Compass or mongosh

### Testing the Flow
1. Register a new user
2. Login with credentials
3. Upload a medical image (test.jpg)
4. Wait for processing (10-30 seconds)
5. View result with side-by-side comparison
6. Check history page for all uploads
7. Click on a log to see detailed stats

## Production Deployment

### Environment Variables
Update these for production:
- `JWT_SECRET` - Use a strong random string
- `MONGODB_URI` - Use MongoDB Atlas or production DB
- `NODE_ENV=production`

### Build Client
```bash
cd client
npm run build
```

### Serve Static Files
Configure backend to serve the built client:
```javascript
app.use(express.static(path.join(__dirname, '../../client/dist')))
```

## File Structure

```
PFE_Test/
├── backend/                 # Node.js + Express
│   ├── src/
│   │   ├── config/         # DB connection
│   │   ├── models/         # Mongoose schemas
│   │   ├── middleware/     # Auth, upload
│   │   ├── controllers/    # Business logic
│   │   ├── routes/         # API routes
│   │   ├── app.js          # Express app
│   │   └── server.js       # Entry point
│   ├── .env                # Environment variables
│   ├── package.json
│   └── Dockerfile
│
├── client/                  # React + Vite
│   ├── src/
│   │   ├── api/            # Axios config
│   │   ├── context/        # Auth context
│   │   ├── pages/          # Route pages
│   │   ├── components/     # Reusable components
│   │   ├── styles/         # CSS
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── Dockerfile
│
├── api/                     # FastAPI (existing)
├── anonymizer/              # AI modules (existing)
├── docker-compose.yml       # Updated with new services
└── MERN_SETUP.md           # This file
```

## Support

For issues or questions:
1. Check logs in terminal/console
2. Verify all services are running
3. Check MongoDB connection
4. Ensure FastAPI pipeline is operational
