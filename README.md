# Deep Learning Image Classification Project

This repository contains a complete deep learning pipeline for image classification, along with a user-friendly web interface for image analysis. The system classifies images into five categories: Painting, Photo, Schematics, Sketch, and Text.

## Table of Contents

- Project Overview
- Repository Structure
- Getting Started
  - Prerequisites
  - Installation
- Deep Learning Pipeline
- Web Interface
- Usage

## Project Overview

This project implements a deep learning-based image classification system that can distinguish between different types of visual content. The model was trained on a dataset of various images and uses the EfficientNetB0 architecture for classification.

## Repository Structure
deep learning/
├── notebooks/
│   └── livrable1/
│       ├── processing/
│       │   └── 5c-v1.ipynb          # Data processing and model training notebook
│       ├── models/
│       │   └── 5cv1/                # Trained models directory
│       │       └── photo_classifier_efficientnetb0_*.h5
│       └── IHM/                     # Web interface
│           ├── app.py               # Flask application
│           ├── static/              # Static assets
│           │   ├── script.js        # Frontend JavaScript
│           │   └── style.css        # CSS styling
│           └── templates/
│               └── index.html       # HTML interface



## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deep-learning.git
   cd deep-learning
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv env
   env\Scripts\activate
   
   # Linux/Mac
   python3 -m venv env
   source env/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install tensorflow opencv-python flask pillow numpy
   ```

## Deep Learning Pipeline

The deep learning pipeline consists of several stages:

1. **Data Preparation**: The notebook 5c-v1.ipynb processes and prepares image data, including:
   - Image validation and repair
   - Resizing to 224×224 pixels
   - Data augmentation
   - Train/validation split

2. **Model Training**: The same notebook contains code for:
   - Model architecture definition (EfficientNetB0)
   - Training configuration
   - Model evaluation
   - Model export to H5 format

3. **Model Selection**: The application automatically selects the most recent model based on the timestamp in the filename.

## Web Interface

The project includes a Flask-based web interface that allows users to:

1. Upload or drag-and-drop images
2. Process images through the trained model
3. View classification results with probability scores
4. Understand the model's confidence in its prediction

### Key Components:

- **Backend** (app.py): Handles model loading, image processing, and API endpoints
- **Frontend** (`index.html`, script.js): Provides an intuitive UI for image upload and result visualization

## Usage

### Running the Web Application

1. Navigate to the IHM directory:
   ```bash
   cd notebooks/livrable1/IHM
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000/`

4. Upload an image using the interface (drag-and-drop or browse)

5. Click "Analyze" to process the image and view the classification results

### API Usage

The system also provides a REST API endpoint for integration with other applications:

- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: Form data with an 'image' file
- **Output**: JSON response with predicted class and probabilities

Example API call using curl:

curl -X POST -F "image=@path/to/your/image.jpg" http://127.0.0.1:5000/predict



## Additional Information

- The system automatically handles image resizing to 224×224 pixels
- The model expects RGB images and will convert other formats as needed
- Maximum supported file size is 5MB

---
