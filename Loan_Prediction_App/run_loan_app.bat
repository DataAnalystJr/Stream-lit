@echo off
echo Starting Loan Prediction Application...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python.
    pause
    exit /b
)

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    
    REM Activate virtual environment
    call venv\Scripts\activate
    
    REM Upgrade pip first
    python -m pip install --upgrade pip
    
    REM Install numpy first to avoid dependency issues
    pip install numpy==1.24.3
    
    REM Install other dependencies
    echo Installing required packages...
    pip install -r requirements.txt
) else (
    REM Just activate the existing virtual environment
    call venv\Scripts\activate
)

REM Run the application
echo Starting Streamlit application...
streamlit run streamlit_app.py

pause 