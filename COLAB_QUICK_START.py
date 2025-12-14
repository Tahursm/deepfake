"""
QUICK START GUIDE FOR GOOGLE COLAB
===================================

This file contains step-by-step instructions for setting up and training
your deepfake detection model in Google Colab.

STEP-BY-STEP PROCESS:
====================

STEP 1: Prepare Your Videos Locally
------------------------------------
1. Make sure your videos are organized in this structure:
   data/
   ├── train/
   │   ├── real/  (your real training videos)
   │   └── fake/  (your fake training videos)
   ├── val/
   │   ├── real/  (your real validation videos)
   │   └── fake/  (your fake validation videos)
   └── test/
       ├── real/  (your real test videos)
       └── fake/  (your fake test videos)

2. Run the upload helper script:
   powershell -ExecutionPolicy Bypass -File upload_to_drive.ps1

STEP 2: Upload Videos to Google Drive
-------------------------------------
1. Go to https://drive.google.com
2. Create a folder called "Deepfake" (or use existing)
3. Inside "Deepfake", create: data/train/real, data/train/fake, etc.
4. Upload your videos to the corresponding folders
   - You can drag and drop from Windows Explorer
   - Or use Google Drive Desktop app for automatic sync

STEP 3: Open Google Colab
-------------------------
1. Go to https://colab.research.google.com
2. Click "New Notebook"
3. Make sure Runtime > Change runtime type > GPU is selected

STEP 4: Clone Your Repository
------------------------------
In Colab, run this in a cell (update with your GitHub URL):
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
!mv YOUR_REPO_NAME/* .
!mv YOUR_REPO_NAME/.* . 2>/dev/null || true

OR if you haven't pushed to GitHub yet:
- Upload your project files using the Files tab (left sidebar)
- Or zip your project and upload it, then unzip

STEP 5: Run the Setup Script
-----------------------------
Copy the code from colab_notebook_steps.py into separate cells:

CELL 1: Install Dependencies
CELL 2: Clone Repository (or skip if you uploaded manually)
CELL 3: Mount Drive and Copy Data
CELL 4: Prepare Metadata
CELL 5: Update Config
CELL 6: Run Training
CELL 7: Download Results

IMPORTANT NOTES:
================

1. Update DRIVE_DATA_PATH in Cell 3:
   Default: '/content/drive/MyDrive/Deepfake/data'
   Change if your folder structure is different

2. Make sure you have videos in all folders before running
   The script will check and warn you if folders are empty

3. Training time depends on:
   - Number of videos
   - Video length
   - Number of epochs
   - GPU availability

4. After training completes:
   - Download the checkpoint files
   - Save them to your local experiments/checkpoints/ folder
   - You can use them for inference on your local machine

TROUBLESHOOTING:
================

Problem: "Data not found" error
Solution: Check that your Drive path is correct and videos are uploaded

Problem: "No samples found" warning
Solution: Make sure videos are in .mp4, .avi, .mov, or .mkv format

Problem: Training fails with CUDA error
Solution: Make sure GPU runtime is enabled in Colab

Problem: Out of memory error
Solution: Reduce batch_size in config (try batch_size: 1)

For more help, check the colab_notebook_steps.py file for detailed code.

"""

# This is just a documentation file - the actual code is in colab_notebook_steps.py

