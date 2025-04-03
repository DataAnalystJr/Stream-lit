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
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies if not already installed
echo Installing required packages...
pip install -r requirements.txt

REM Run the application
echo Starting Streamlit application...
streamlit run streamlit_app.py

pause 