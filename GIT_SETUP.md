# Git Setup and GitHub Push Instructions

## Step 1: Initialize Local Git Repository

```bash
# Navigate to project directory
cd c:\Users\MSI\Desktop\PFE_Test

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit - Medical Image Anonymization Pipeline"
```

## Step 2: Add Docker Support Commit

```bash
# Stage Docker files
git add Dockerfile docker-compose.yml .dockerignore

# Commit Docker configuration
git commit -m "Add Docker support for containerized deployment"
```

## Step 3: Add Documentation Commit

```bash
# Stage documentation
git add README.md PROJECT_REPORT.md requirements.txt

# Commit documentation
git commit -m "Add comprehensive documentation and project report"
```

## Step 4: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to https://github.com/new
2. Repository name: `medical-image-anonymizer`
3. Description: `AI-powered medical image anonymization with CLIP classification and dual OCR`
4. Visibility: Public or Private (your choice)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
# Install GitHub CLI first: https://cli.github.com/
gh repo create medical-image-anonymizer --public --source=. --remote=origin
```

## Step 5: Connect Local Repository to GitHub

```bash
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/medical-image-anonymizer.git

# Verify remote
git remote -v
```

## Step 6: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 7: Verify Upload

Visit: `https://github.com/YOUR_USERNAME/medical-image-anonymizer`

You should see:
- ✓ All Python files
- ✓ Dockerfile and docker-compose.yml
- ✓ README.md displayed on homepage
- ✓ requirements.txt
- ✓ .gitignore and .dockerignore

**NOT included** (per .gitignore):
- ✗ venv/ directory
- ✗ __pycache__/ directories
- ✗ output/ directory
- ✗ *.pyc files

## Complete Command Sequence (Copy-Paste Ready)

```bash
# 1. Initialize repository
cd c:\Users\MSI\Desktop\PFE_Test
git init
git add .
git commit -m "Initial commit - Medical Image Anonymization Pipeline"

# 2. Add Docker support
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "Add Docker support for containerized deployment"

# 3. Add documentation
git add README.md PROJECT_REPORT.md requirements.txt
git commit -m "Add comprehensive documentation and project report"

# 4. Connect to GitHub (REPLACE YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/medical-image-anonymizer.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

## For Your Advisor

After pushing to GitHub, share this URL with your advisor:

```
https://github.com/YOUR_USERNAME/medical-image-anonymizer
```

They can then:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-image-anonymizer.git
cd medical-image-anonymizer
```

### 2. Build Docker image
```bash
docker build -t medical-anonymizer .
```

### 3. Run the pipeline
```bash
# Create test directories
mkdir -p input output

# Place test image in input/
cp /path/to/test_image.jpg input/

# Run pipeline
docker run -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           medical-anonymizer input/test_image.jpg output/
```

### 4. View results
```bash
ls output/
# Output: anonymized_test_image.jpg
```

## Troubleshooting

### Issue: "fatal: not a git repository"
**Solution:** Run `git init` first

### Issue: "remote origin already exists"
**Solution:** 
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/medical-image-anonymizer.git
```

### Issue: "failed to push some refs"
**Solution:**
```bash
# Pull first if repository has content
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Issue: Authentication failed
**Solution:** Use Personal Access Token instead of password
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when prompted

## Alternative: Using SSH

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub
# Copy public key: cat ~/.ssh/id_ed25519.pub
# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Use SSH remote instead
git remote add origin git@github.com:YOUR_USERNAME/medical-image-anonymizer.git
git push -u origin main
```

## Repository Settings Recommendations

After pushing, configure these on GitHub:

1. **Add Topics/Tags:**
   - medical-imaging
   - anonymization
   - ocr
   - deep-learning
   - clip
   - docker
   - python

2. **Add Description:**
   ```
   AI-powered medical image anonymization pipeline with CLIP classification, 
   dual OCR (PaddleOCR + EasyOCR), and safe pixel redaction. Docker-ready.
   ```

3. **Enable Issues:** For advisor feedback

4. **Add License:** MIT (if open source)

5. **Create Releases:** Tag stable versions
   ```bash
   git tag -a v1.0.0 -m "Initial release - Complete 7-stage pipeline"
   git push origin v1.0.0
   ```

## Success Checklist

- [ ] Repository initialized locally
- [ ] All commits created
- [ ] GitHub repository created
- [ ] Remote origin added
- [ ] Code pushed successfully
- [ ] README.md displays on GitHub homepage
- [ ] Dockerfile and docker-compose.yml visible
- [ ] .gitignore working (venv/ not uploaded)
- [ ] Repository URL shared with advisor
- [ ] Docker build tested by advisor

---

**Ready to share with your advisor!** 🎉
