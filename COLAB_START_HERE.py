"""
🚀 QUICK START: RUN PROJECT IN GOOGLE COLAB FROM GITHUB
=======================================================

STEP-BY-STEP INSTRUCTIONS:
==========================

1. Open Google Colab
   - Go to: https://colab.research.google.com
   - Click "New Notebook"

2. Enable GPU (IMPORTANT!)
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

3. Open the file: colab_complete_setup.py
   - This file contains all the cells you need
   - Copy each CELL section into a separate cell in Colab
   - Run them in order (Cell 1, Cell 2, Cell 3, etc.)

OR use the simpler version:

3. Open the file: colab_notebook_steps.py
   - Copy each CELL section into separate cells
   - Run them in order

CELL BREAKDOWN:
===============

CELL 1: Install Dependencies
  - Installs all required Python packages
  - Takes 2-5 minutes

CELL 2: Clone from GitHub
  - Clones your repository from GitHub
  - Moves files to Colab workspace
  - Already configured for: https://github.com/Tahursm/deepfake.git

CELL 3: Mount Google Drive & Copy Data
  - Mounts your Google Drive
  - Creates data folder structure
  - Copies videos from Drive to Colab
  - IMPORTANT: Upload your videos to Drive first!
    Path: /content/drive/MyDrive/Deepfake/data/

CELL 4: Prepare Dataset Metadata
  - Creates metadata JSON files for train/val/test
  - Scans video folders and creates labels

CELL 5: Update Config for Colab
  - Sets batch_size=2 for GPU
  - Sets num_workers=0 (prevents segfault)

CELL 6: Verify Setup
  - Checks that everything is ready
  - Shows data summary

CELL 7: Run Training
  - Starts the training process
  - This takes the longest time

CELL 8: Download Results
  - Downloads trained model checkpoints
  - Downloads training logs

IMPORTANT NOTES:
================

1. Make sure GPU is enabled before starting
2. Upload videos to Google Drive BEFORE running Cell 3
3. The repository URL is already set to: https://github.com/Tahursm/deepfake.git
4. If you get errors, check that:
   - GPU is enabled
   - Videos are uploaded to Drive
   - All cells are run in order

TROUBLESHOOTING:
================

Problem: "Repository not found"
Solution: Check that the GitHub URL is correct and the repo is public

Problem: "Data not found"
Solution: Upload videos to Google Drive at: MyDrive/Deepfake/data/

Problem: "Segmentation fault"
Solution: Already fixed! num_workers=0 is set automatically

Problem: "Out of memory"
Solution: Reduce batch_size in config to 1

For detailed code, see: colab_complete_setup.py or colab_notebook_steps.py

"""

