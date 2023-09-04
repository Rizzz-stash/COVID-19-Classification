# COVID-19 Classification

This repository contains my final year project, which utilizes MobileNet architecture to identify COVID-19.

## Introduction

This research makes a significant contribution to the advancement of knowledge and methodologies in the fields of pattern recognition and image processing. The research outcomes provide a deeper understanding of the use of CNN with MobileNet architecture using transfer learning for COVID-19 image classification. It is worth noting that, by utilizing the MobileNet architecture, this model becomes highly lightweight for mobile applications or deployment.

## Features

- **COVID-19 Image Classification:** Utilize a Convolutional Neural Network (CNN) with MobileNet architecture to classify COVID-19 images accurately.
- **High Accuracy:** Achieve high accuracy in the classification of COVID-19 images, ensuring reliable results.
- **Continuous Improvement:** Commitment to ongoing development and improvement of the COVID-19 classification model.

## Technologies Used

- **TensorFlow:** The core machine learning library used for building and training the COVID-19 classification model.

- **Python:** The primary programming language for developing the project's code and scripts.

- **OpenCV:** Used for image preprocessing and handling in the COVID-19 image classification pipeline.

- **Matplotlib:** Utilized for plotting and visualizing training results, providing insights into model performance.

- **CUDA (Attempted):** An attempt was made to leverage CUDA for GPU acceleration, but due to GPU limitations, it was not successfully implemented.

## Dataset

The research dataset comprises chest X-ray images, encompassing three distinct patient categories. These categories consist of 975 individuals diagnosed with COVID-19, 988 X-ray images representing healthy patients, and 989 X-ray images depicting patients afflicted by Viral Pneumonia. The dataset was sourced from the [Balanced Augmented Covid CXR Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/balanced-augmented-covid-cxr-dataset) available on Kaggle. To conform to MobileNet's prerequisites, all X-ray images within the datasets were standardized to a uniform size of 224 × 224 pixels.

