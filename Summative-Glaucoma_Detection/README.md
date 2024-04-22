# Glaucoma Detection with Deep Learning

This project aims to build an automated system for detecting Glaucoma, an eye condition that can lead to vision loss, using deep learning techniques. The system takes retinal scan images as input and predicts whether a person is affected by Glaucoma or not.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Deployment](#model-deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Problem Statement

Glaucoma is a group of eye diseases that cause progressive damage to the optic nerve, leading to vision loss and potential blindness if left untreated. Early detection and treatment are crucial for preventing further vision loss. However, manual diagnosis by ophthalmologists can be time-consuming, subjective, and prone to human error.

This project addresses the need for an accurate and efficient automated system for Glaucoma detection by leveraging deep learning techniques and retinal scan images.

## Dataset

The dataset used in this project consists of retinal scan images from patients, both with and without Glaucoma. The images are divided into training, validation, and test sets for model training and evaluation.

## Approach

The project follows a deep learning approach using transfer learning and the EfficientNetB3 model as the base model. The model is trained on the retinal scan images to classify them into two categories: "Normal" (no Glaucoma) and "Glaucoma."

The main steps involved in the project are:

1. Data Preprocessing
2. Data Augmentation
3. Model Architecture and Training
4. Model Evaluation
5. Model Deployment

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your-username/glaucoma-detection.git
```

2. Install the required packages:
```pip install -r requirements.txt```

