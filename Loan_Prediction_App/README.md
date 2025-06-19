# Loan Approval Prediction Application

This is a Streamlit-based web application that predicts whether a loan applicant is likely to repay their loan based on various factors such as income, credit history, education, and other relevant parameters.

## Features

- Interactive web interface for loan application input
- Real-time loan approval prediction
- Visual representation of prediction probabilities
- Support for various input parameters including:
  - Personal information (gender, marital status, dependents)
  - Financial information (income, loan amount, loan term)
  - Credit history and property details
  - Education and employment status

## System Requirements

- Python 3.8 or higher
- Windows operating system (for the batch file)
- Internet connection (for first-time package installation)

## Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Method 1: Using the Batch File (Windows)
Simply double-click the `run_loan_app.bat` file in the application directory.

### Method 2: Using Command Line
1. Open a terminal in the application directory
2. Run the following command:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Fill in all the required fields in the application form
2. Click the "Submit" button to get the prediction
3. View the prediction results and probability visualization
4. Use the "Clear" button to reset the form for a new prediction

## Model Information

The application uses a Random Forest model trained on historical loan data. The model considers various features to make predictions about loan repayment probability.

## Support

If you encounter any issues or have questions, please check the following:
1. Ensure all required packages are installed correctly
2. Verify that Python 3.8 or higher is installed
3. Check that all files are in their correct locations

## License

This project is licensed under the terms included in the LICENSE file. 