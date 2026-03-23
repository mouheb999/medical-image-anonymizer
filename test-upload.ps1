# Test script for medical image anonymization

Write-Host "=== Medical Image Anonymizer Test ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Register a test user
Write-Host "1. Registering test user..." -ForegroundColor Yellow
$registerBody = @{
    name = "Test User"
    email = "test@example.com"
    password = "password123"
} | ConvertTo-Json

try {
    $registerResponse = Invoke-RestMethod -Uri "http://localhost:5000/api/auth/register" `
        -Method POST `
        -ContentType "application/json" `
        -Body $registerBody
    
    $token = $registerResponse.token
    Write-Host "✓ User registered successfully" -ForegroundColor Green
    Write-Host "  Token: $($token.Substring(0, 20))..." -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "✗ Registration failed (user may already exist)" -ForegroundColor Red
    
    try {
        # Try to login instead
        Write-Host "  Attempting to login..." -ForegroundColor Yellow
        $loginBody = @{
            email = "test@example.com"
            password = "password123"
        } | ConvertTo-Json
        
        $loginResponse = Invoke-RestMethod -Uri "http://localhost:5000/api/auth/login" `
            -Method POST `
            -ContentType "application/json" `
            -Body $loginBody
        
        $token = $loginResponse.token
        Write-Host "✓ Login successful" -ForegroundColor Green
        Write-Host ""
    } catch {
        Write-Host "✗ Login also failed" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Upload and anonymize image
Write-Host "2. Uploading image for anonymization..." -ForegroundColor Yellow

$imagePath = "person49_virus_101.jpeg"
if (-not (Test-Path $imagePath)) {
    Write-Host "✗ Error: Image file not found: $imagePath" -ForegroundColor Red
    Write-Host "  Please make sure the image is in the current directory" -ForegroundColor Yellow
    exit 1
}

try {
    $boundary = [System.Guid]::NewGuid().ToString()
    $fileBytes = [System.IO.File]::ReadAllBytes($imagePath)
    $fileContent = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes)
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"file`"; filename=`"$imagePath`"",
        "Content-Type: image/jpeg",
        "",
        $fileContent,
        "--$boundary--"
    ) -join "`r`n"
    
    Write-Host "  Sending to FastAPI pipeline..." -ForegroundColor Gray
    
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/anonymize" `
        -Method POST `
        -Headers @{
            "Authorization" = "Bearer $token"
            "Content-Type" = "multipart/form-data; boundary=$boundary"
        } `
        -Body $bodyLines
    
    Write-Host "✓ Anonymization complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "=== Results ===" -ForegroundColor Cyan
    Write-Host "  Classification: $($response.classification)" -ForegroundColor White
    Write-Host "  Confidence: $([math]::Round($response.confidence * 100, 2))%" -ForegroundColor White
    Write-Host "  Format: $($response.format)" -ForegroundColor White
    Write-Host "  Regions Detected: $($response.total_regions)" -ForegroundColor White
    Write-Host "  Regions Redacted: $($response.redacted)" -ForegroundColor White
    Write-Host "  Processing Time: $([math]::Round($response.processingTime / 1000, 2))s" -ForegroundColor White
    Write-Host "  Output File: $($response.output_filename)" -ForegroundColor White
    Write-Host ""
    Write-Host "✓ Download from: http://localhost:8000/result/$($response.output_filename)" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Anonymization failed" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.ErrorDetails.Message) {
        $errorObj = $_.ErrorDetails.Message | ConvertFrom-Json
        Write-Host "  Details: $($errorObj.message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
