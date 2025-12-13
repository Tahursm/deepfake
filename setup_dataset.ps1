# Dataset Setup Script for Windows PowerShell
# Creates the directory structure for organizing deepfake detection dataset

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset Directory Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create directory structure
$splits = @("train", "val", "test")
$classes = @("real", "fake")

$created = 0
foreach ($split in $splits) {
    foreach ($class in $classes) {
        $path = "data\$split\$class"
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Force -Path $path | Out-Null
            Write-Host "Created: $path" -ForegroundColor Green
            $created++
        } else {
            Write-Host "Exists: $path" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Directory structure ready!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Instructions
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy your REAL videos to:" -ForegroundColor White
Write-Host "   - data\train\real\" -ForegroundColor Gray
Write-Host "   - data\val\real\" -ForegroundColor Gray
Write-Host "   - data\test\real\" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Copy your FAKE videos to:" -ForegroundColor White
Write-Host "   - data\train\fake\" -ForegroundColor Gray
Write-Host "   - data\val\fake\" -ForegroundColor Gray
Write-Host "   - data\test\fake\" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Generate metadata:" -ForegroundColor White
Write-Host "   python scripts/prepare_dataset.py --data_dir data --splits train val test" -ForegroundColor Cyan
Write-Host ""

# Check if videos already exist
$totalVideos = 0
foreach ($split in $splits) {
    foreach ($class in $classes) {
        $path = "data\$split\$class"
        if (Test-Path $path) {
            $videos = Get-ChildItem -Path $path -Filter "*.mp4" -ErrorAction SilentlyContinue
            $count = $videos.Count
            $totalVideos += $count
            if ($count -gt 0) {
                Write-Host "Found $count videos in $path" -ForegroundColor Green
            }
        }
    }
}

if ($totalVideos -gt 0) {
    Write-Host ""
    Write-Host "Total videos found: $totalVideos" -ForegroundColor Cyan
    Write-Host "You can now generate metadata!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "No videos found yet. Add videos to the directories above." -ForegroundColor Yellow
}

