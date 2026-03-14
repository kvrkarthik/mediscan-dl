# 🔬 MediScan AI — Deep Learning Edition

## Architecture
- **Chest X-Ray**: ViT-Base-Patch16-224 (MedMNIST ChestMNIST)
- **Skin Lesion**: EfficientNet-B4 (ISIC Dataset)
- **Retina**: ViT-Base-Patch16-224 (MedMNIST RetinaMNIST)
- **Tissue/MRI**: ViT-Base-Patch16-224 (MedMNIST TissueMNIST)
- **Blood Cells**: ViT-Base-Patch16-224 (MedMNIST BloodMNIST)

## Setup

### Step 1 — Get HuggingFace API Token (Free)
1. Go to https://huggingface.co and sign up
2. Click your profile → Settings → Access Tokens
3. Click "New Token" → name it "mediscan" → Role: Read
4. Copy the token (starts with `hf_...`)

### Step 2 — Push to GitHub
```bash
git init
git add .
git commit -m "MediScan DL version"
git branch -M main
git remote add origin https://github.com/kvrkarthik/mediscan-ai.git
git push -u origin main --force
```

### Step 3 — Deploy on Vercel
1. Go to vercel.com → Import your GitHub repo
2. Add Environment Variable:
   - Name: `HF_API_KEY`
   - Value: your HuggingFace token
3. Click Deploy ✅

## Note on Model Loading
HuggingFace free tier models "sleep" after inactivity.
First request may take 20-30 seconds to wake up.
Subsequent requests are fast. This is normal behavior.