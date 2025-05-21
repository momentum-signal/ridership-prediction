# Ridership and High/Low Traffic Prediction

## Project Overview

The goal of this project is to analyze and predict **ridership data** for **Komuter** services in Malaysia. The focus is on predicting **high** and **low traffic periods** based on historical ridership patterns. This can help in better planning, optimizing resources, and improving the commuter experience.

### Key Objectives
1. **Ridership Analysis**: Analyze the ridership data from various origins to destinations, focusing on peak and off-peak times.
2. **Traffic Prediction**: Use machine learning techniques to predict periods of high and low ridership. This will be useful for authorities and commuters to better understand travel patterns and prepare accordingly.
3. **Visualization**: Generate visualizations (such as graphs, charts) to represent the trends in ridership and highlight peak traffic periods.

## Data Source

The data for this project is sourced from the official Malaysian government portal for **Komuter Ridership**:

- **Dataset Link**: [Hourly Origin-Destination Ridership: Komuter](https://data.gov.my/data-catalogue/ridership_od_komuter?)


## Prerequisites

- Python 3.9+
- pip 20+

## Local Setup
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ridership-prediction.git
cd ridership-prediction
```

### 2. Set up virtual environment
``` bash
python -m venv .venv

# Activate environment:
# Linux/Mac:
source .venv/bin/activate  
# Windows:
.\.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Directory structure
```bash
├──model
    ├── data/
    │   └── cleaned_data.csv
    ├── models/      
    │   ├── models/            # Model definitions
    │   ├── saved_models/      # Trained models
    │   └── utils/             # Data processing
    ├── main.py/               # FastAPI app
    ├── data_processing.py/    # Data processing
    ├── requirements.txt
    └── README.md
```

## Running the Project
### 1. Train the model
```bash
python -m model.train  # Or run the training notebook
```

### 2. Start the API Server
``` bash
uvicorn model.main:app --reload
```

