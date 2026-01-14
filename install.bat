@echo off
echo ==========================================================
echo    Setting up environment for V3S_GIA (Windows Fixed)
echo ==========================================================

:: 1. Create Conda environment
echo [1/3] Checking Conda environment 'V3S_GIA'...
call conda create -n V3S_GIA python=3.10 -y

:: 2. Activate environment (????: ?? conda.bat activate)
echo.
echo Activating environment...
call conda.bat activate V3S_GIA

:: ????????
if errorlevel 1 (
   echo.
   echo [ERROR] Activation failed! 
   echo Please run "conda init cmd.exe" in your terminal once, restart console, and try again.
   pause
   exit /b
)

:: 3. Install PyTorch (?????)
echo.
echo [2/3] Installing PyTorch 2.1.0 (CUDA 12.1)...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

:: 4. Install requirements
echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo ==========================================================
echo    Setup Finished! Please run: conda activate V3S_GIA
echo ==========================================================
pause
