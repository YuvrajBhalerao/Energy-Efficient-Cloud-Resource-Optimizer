Energy-Efficient Cloud Resource Optimizer
!

This project is a Python-based application that uses machine learning to predict resource usage (CPU, GPU, memory) in a cloud environment. Based on these predictions, it dynamically recommends resource allocation strategies to minimize energy consumption and reduce operational costs. It includes a simulator to quantify potential savings in terms of cost and Azure cloud credits.

🚀 Key Features
ML-Powered Prediction: Utilizes a machine learning model (Random Forest) to forecast future resource needs based on historical telemetry data.

Dynamic Resource Allocation: Implements a rule-based engine to suggest scaling actions (SCALE_UP, SCALE_DOWN, MAINTAIN).

Cost & Energy Simulation: Simulates the financial and environmental impact of the optimization strategy, providing clear metrics on savings.

Flask API: Exposes a simple REST API to trigger the optimization pipeline and retrieve results.

Ready for Deployment: Configured for seamless deployment to the cloud using Render.

📂 Project Structure
.
├── data/
│   └── sample_metrics.csv      # Sample telemetry data
├── src/
│   └── optimizer/
│       ├── __init__.py
│       ├── data_loader.py        # Handles loading & preprocessing
│       ├── feature_engineer.py   # Feature extraction for ML
│       ├── model.py              # ML model to predict usage
│       ├── resource_allocator.py # Dynamic resource allocation logic
│       └── simulator.py          # Simulates optimization results
├── templates/
│   └── index.html              # Frontend UI for visualization
├── .gitignore
├── LICENSE
├── Procfile                    # Deployment command for Render
├── README.md
├── app.py                      # Flask application entry point
├── render.yaml                 # Render deployment configuration
├── requirements.txt            # Python dependencies
└── runtime.txt                 # Specifies python-3.12.5

🛠️ Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

Prerequisites
Python 3.12.5

pip (Python package installer)

Installation & Local Setup
Clone the repository:

git clone <your-repository-url>
cd <repository-directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Run the application:

python app.py

The application will start a local server, typically on http://127.0.0.1:5000. The first time you run it, it will generate a data/sample_metrics.csv file if one is not found.

☁️ Deployment to Render
This project is configured for easy deployment on Render.

Push your code to a GitHub repository.

Go to the Render Dashboard and create a new Web Service.

Connect the GitHub repository you just created.

Render will automatically detect the render.yaml file and configure the settings for you.

Service Name: energy-efficient-cloud-optimizer (or your choice)

Runtime: Python

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Click Create Web Service. Render will build and deploy your application. Your service will be live at the URL provided by Render.

🔌 API Endpoint
The application exposes one main API endpoint to run the entire optimization process.

POST /api/run-optimization
Triggers the full pipeline: loading data, engineering features, making predictions, and running the cost-saving simulation.

Method: POST

Body: None

Success Response (200 OK):

{
  "status": "success",
  "data": {
    "total_intervals": 162,
    "original_cost": 97.20,
    "optimized_cost": 74.25,
    "cost_savings": 22.95,
    "cost_savings_percent": 23.61,
    "energy_saved_kwh": 70.51
  }
}

Error Response (404 Not Found or 500 Internal Server Error):

{
    "status": "error",
    "message": "Error message describing the issue."
}
