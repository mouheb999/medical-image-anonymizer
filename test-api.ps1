Write-Host "=== Medical Image Anonymizer API Test ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Register user
Write-Host "Step 1: Registering user..." -ForegroundColor Yellow
$registerData = @{
    name = "Test User"
    email = "test@example.com"
    password = "password123"
}

$token = $null

try {
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/auth/register" -Method POST -Body ($registerData | ConvertTo-Json) -ContentType "application/json"
    $token = $response.token
    Write-Host "SUCCESS: User registered" -ForegroundColor Green
} catch {
    Write-Host "User already exists, trying login..." -ForegroundColor Yellow
    $loginData = @{
        email = "test@example.com"
        password = "password123"
    }
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/auth/login" -Method POST -Body ($loginData | ConvertTo-Json) -ContentType "application/json"
    $token = $response.token
    Write-Host "SUCCESS: Logged in" -ForegroundColor Green
}

Write-Host "Token: $token" -ForegroundColor Gray
Write-Host ""

# Step 2: Check if image exists
Write-Host "Step 2: Checking for image file..." -ForegroundColor Yellow
$imagePath = "person49_virus_101.jpeg"

if (-not (Test-Path $imagePath)) {
    Write-Host "ERROR: Image not found: $imagePath" -ForegroundColor Red
    Write-Host "Please make sure the image is in: $PWD" -ForegroundColor Yellow
    exit 1
}

Write-Host "SUCCESS: Image found" -ForegroundColor Green
Write-Host ""

# Step 3: Upload image
Write-Host "Step 3: Uploading image to anonymize..." -ForegroundColor Yellow
Write-Host "This will take 10-30 seconds..." -ForegroundColor Gray

$uri = "http://localhost:5000/api/anonymize"
$headers = @{
    "Authorization" = "Bearer $token"
}

# Use curl.exe (the real curl, not PowerShell alias)
$curlPath = "C:\Windows\System32\curl.exe"

if (Test-Path $curlPath) {
    Write-Host "Using curl.exe..." -ForegroundColor Gray
    & $curlPath -X POST $uri -H "Authorization: Bearer $token" -F "file=@$imagePath"
} else {
    Write-Host "curl.exe not found, using Invoke-WebRequest..." -ForegroundColor Gray
    
    $fileBin = [System.IO.File]::ReadAllBytes((Resolve-Path $imagePath))
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $bodyLines = (
        "--$boundary",
        "Content-Disposition: form-data; name=`"file`"; filename=`"$imagePath`"",
        "Content-Type: image/jpeg$LF",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBin),
        "--$boundary--$LF"
    ) -join $LF
    
    $headers["Content-Type"] = "multipart/form-data; boundary=$boundary"
    
    try {
        $result = Invoke-RestMethod -Uri $uri -Method POST -Headers $headers -Body $bodyLines
        Write-Host ""
        Write-Host "SUCCESS: Anonymization complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== Results ===" -ForegroundColor Cyan
        Write-Host "Classification: $($result.classification)" -ForegroundColor White
        Write-Host "Confidence: $([math]::Round($result.confidence * 100, 2))%" -ForegroundColor White
        Write-Host "Regions Redacted: $($result.redacted)" -ForegroundColor White
        Write-Host "Processing Time: $([math]::Round($result.processingTime / 1000, 2))s" -ForegroundColor White
        Write-Host "Output: $($result.output_filename)" -ForegroundColor White
    } catch {
        Write-Host ""
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
