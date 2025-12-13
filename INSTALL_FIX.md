# Quick Fix for Current Errors (Windows)

> **Windows OS Optimized** - All commands are for Windows PowerShell.

## Issue 1: Protobuf Dependency Conflict

The warning about `mysql-connector-python` and `protobuf` is usually non-critical, but here's how to fix it:

### Option A: Uninstall mysql-connector-python (Recommended)
```powershell
# Activate virtual environment first
.venv\Scripts\Activate.ps1

# Uninstall the conflicting package
pip uninstall mysql-connector-python -y
```

### Option B: Run the fix script
```powershell
python scripts/fix_dependencies.py
```

### Option C: Install compatible protobuf
```powershell
pip install 'protobuf>=4.21.0,<4.26.0'
```

---

## Issue 2: Command Path Error

**Wrong command (has typo):**
```powershell
0.e:\Deepfake\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Correct commands:**

### Method 1: Activate virtual environment first (Recommended)
```powershell
# Navigate to project directory
cd E:\Deepfake

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Now you can use 'python' directly
python -m pip install -r requirements.txt
```

### Method 2: Use full correct path
```powershell
E:\Deepfake\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Method 3: Use relative path from project root
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Complete Installation Steps

1. **Navigate to project:**
   ```powershell
   cd E:\Deepfake
   ```

2. **Activate virtual environment:**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   (You should see `(.venv)` in your prompt)

3. **Fix protobuf conflict (if needed):**
   ```powershell
   pip uninstall mysql-connector-python -y
   ```

4. **Install/upgrade requirements:**
   ```powershell
   pip install -r requirements.txt --upgrade
   ```

5. **Verify installation:**
   ```powershell
   python -c "import torch; import mediapipe; print('Installation successful!')"
   ```

---

## If Virtual Environment Activation Fails

If you get an execution policy error in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.venv\Scripts\Activate.ps1
```

---

## Note About Warnings

The protobuf warning is usually **non-critical**. Your packages should work fine despite the warning. The warning appears because:
- `mysql-connector-python` (if installed) wants an older protobuf version
- Other packages (like MediaPipe, TensorFlow) want a newer protobuf version

Since this project doesn't use `mysql-connector-python`, you can safely ignore or uninstall it.

