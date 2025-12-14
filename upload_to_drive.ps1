# PowerShell script to help organize videos for Google Drive upload
# This script will help you prepare your videos for upload to Google Drive

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Google Drive Upload Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if data directory exists
$dataDir = "data"
if (-not (Test-Path $dataDir)) {
    Write-Host "❌ Data directory not found at: $dataDir" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}

Write-Host "📁 Checking your local data structure..." -ForegroundColor Yellow
Write-Host ""

# Count videos in each folder
$totalVideos = 0
foreach ($split in @('train', 'val', 'test')) {
    foreach ($label in @('real', 'fake')) {
        $folderPath = Join-Path $dataDir $split $label
        if (Test-Path $folderPath) {
            $videos = Get-ChildItem -Path $folderPath -Include *.mp4,*.avi,*.mov,*.mkv -Recurse
            $count = $videos.Count
            $totalVideos += $count
            Write-Host "  $split/$label : $count videos" -ForegroundColor Green
        } else {
            Write-Host "  $split/$label : Folder not found" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Total videos found: $totalVideos" -ForegroundColor Cyan
Write-Host ""

# Instructions
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📋 UPLOAD INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open Google Drive in your web browser" -ForegroundColor White
Write-Host "   https://drive.google.com" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Create a folder called 'Deepfake' in your Drive" -ForegroundColor White
Write-Host "   (or use an existing folder)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Inside 'Deepfake', create this folder structure:" -ForegroundColor White
Write-Host "   Deepfake/" -ForegroundColor Gray
Write-Host "   └── data/" -ForegroundColor Gray
Write-Host "       ├── train/" -ForegroundColor Gray
Write-Host "       │   ├── real/" -ForegroundColor Gray
Write-Host "       │   └── fake/" -ForegroundColor Gray
Write-Host "       ├── val/" -ForegroundColor Gray
Write-Host "       │   ├── real/" -ForegroundColor Gray
Write-Host "       │   └── fake/" -ForegroundColor Gray
Write-Host "       └── test/" -ForegroundColor Gray
Write-Host "           ├── real/" -ForegroundColor Gray
Write-Host "           └── fake/" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Upload your videos:" -ForegroundColor White
Write-Host ""
Write-Host "   Option A: Manual Upload (Recommended for first time)" -ForegroundColor Yellow
Write-Host "   - Open each folder (data/train/real, data/train/fake, etc.)" -ForegroundColor Gray
Write-Host "   - Drag and drop videos from your local folders to Drive folders" -ForegroundColor Gray
Write-Host ""
Write-Host "   Option B: Use Google Drive Desktop App" -ForegroundColor Yellow
Write-Host "   - Install Google Drive for Desktop" -ForegroundColor Gray
Write-Host "   - Sync your local 'data' folder to Drive/Deepfake/data" -ForegroundColor Gray
Write-Host "   - This will automatically upload all videos" -ForegroundColor Gray
Write-Host ""
Write-Host "   Option C: Use rclone (Advanced)" -ForegroundColor Yellow
Write-Host "   - Install rclone and configure Google Drive" -ForegroundColor Gray
Write-Host "   - Use: rclone copy data/ remote:Deepfake/data/" -ForegroundColor Gray
Write-Host ""

# Ask if user wants to open the folders
$response = Read-Host "Would you like to open your local data folders? (Y/N)"
if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host ""
    Write-Host "Opening data folders..." -ForegroundColor Yellow
    
    foreach ($split in @('train', 'val', 'test')) {
        foreach ($label in @('real', 'fake')) {
            $folderPath = Join-Path $dataDir $split $label
            if (Test-Path $folderPath) {
                Start-Process explorer.exe -ArgumentList $folderPath
                Start-Sleep -Milliseconds 500
            }
        }
    }
    
    Write-Host "✅ Folders opened!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. After uploading videos to Google Drive," -ForegroundColor White
Write-Host "2. Open Google Colab (https://colab.research.google.com)" -ForegroundColor White
Write-Host "3. Create a new notebook" -ForegroundColor White
Write-Host "4. Copy the code from 'colab_notebook_steps.py' into separate cells" -ForegroundColor White
Write-Host "5. Update the DRIVE_DATA_PATH in Cell 3 to match your Drive path" -ForegroundColor White
Write-Host "6. Run all cells!" -ForegroundColor White
Write-Host ""
Write-Host "Your Drive path should be:" -ForegroundColor Yellow
Write-Host "  /content/drive/MyDrive/Deepfake/data" -ForegroundColor Gray
Write-Host ""
Write-Host "Or if you used a different folder name:" -ForegroundColor Yellow
Write-Host "  /content/drive/MyDrive/YOUR_FOLDER_NAME/data" -ForegroundColor Gray
Write-Host ""

