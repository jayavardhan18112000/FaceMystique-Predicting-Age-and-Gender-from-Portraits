# FaceMystique-Predicting-Age-and-Gender-from-Portraits

## Overview
This project showcases a deep learning approach to predict the gender and approximate age of individuals from facial images. Leveraging the power of convolutional neural networks (CNNs), this model is designed to provide accurate gender and age predictions.

![Example Prediction](example.png)

## Dataset
The project utilizes the UTKFace dataset, a rich resource of grayscale facial images, each annotated with age and gender labels. This dataset is publicly available on Kaggle and can be downloaded [here](https://www.kaggle.com/jangedoo/utkface-new).

## Model Architecture
The heart of this project is a well-crafted deep learning model with the following architecture:

- **Convolutional Layers**: Multiple convolutional layers to automatically extract meaningful features from facial images.
- **Pooling Layers**: Max-pooling layers to downsample the feature maps.
- **Fully Connected Layers**: Dense layers for high-level feature aggregation.
- **Output Branches**: The model has two output branches:
  - Gender Prediction: Utilizes binary cross-entropy loss for accurate gender classification.
  - Age Prediction: Uses mean absolute error (MAE) loss to approximate the age of the individual.

The use of two output branches allows the model to simultaneously predict both gender and age, providing a comprehensive analysis of facial data.

## Training
The model is trained using a carefully curated dataset split into training and validation sets. The training process includes the following key details:

- **Loss Functions**: Binary cross-entropy for gender prediction and MAE for age prediction.
- **Optimizer**: The Adam optimizer is employed to efficiently update model weights.
- **Batch Size**: Training is conducted in batches of 32 samples.
- **Epochs**: The model undergoes 30 epochs to ensure robust learning.

![epochs](epoch_results_1.png)
![epochs](epoch_results_2.png)
![epochs](epoch_results_3.png)
![epochs](epoch_results_5.png)

## Results
The performance of the trained model is evaluated on both gender and age prediction tasks. The model's accuracy and loss metrics are visualized over training epochs, allowing for an in-depth analysis of its learning process.

![results](result_1.png)
![results](result_2.png)
![results](result_3.png)
![results](result_4.png)

## Usage
You can run this project directly in a Kaggle notebook. Follow these steps:

1. Open the Kaggle notebook associated with this project.
2. Ensure access to the UTKFace dataset on Kaggle or upload it if necessary.
3. Execute the notebook cells one by one to load data, build and train the model, and make predictions.

## Requirements
Ensure you have the following dependencies installed:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- tqdm
