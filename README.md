# Anomaly Detection System

## Description
A machine learning-based anomaly detection system using deep learning to identify anomalies in images/documents.

## Features
- PDF to image conversion
- Automated anomaly detection
- Flask web interface
- Model training capabilities
- Web scraping for data collection

## Setup

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure
├── app.py                          # Main Flask application
├── train.py                        # Model training script
├── preprocess.py                   # Data preprocessing
├── pdf2img.py                      # PDF conversion utility
├── scraping.py                     # Web scraping script
├── anomaly_detection_model.h5      # Trained model (not in repo)
├── templates/
│   └── index.html                  # Web interface
├── uploads/                        # Upload directory
├── anomalies/                      # Detected anomalies
└── normal/                         # Normal samples

## Model
The system uses a deep learning model trained on [describe your dataset].

## License
[Your License]
# Anomalous-Election-Result-Checker
