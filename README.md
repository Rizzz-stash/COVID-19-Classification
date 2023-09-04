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

## Conclusion

In summary, this research has successfully achieved its established objectives:

1. **Classification Model with CNN MobileNet:**
   This study effectively implemented the Convolutional Neural Network (CNN) method using MobileNet architecture for the classification of chest X-ray images. Through transfer learning, the model successfully categorized these images into three main categories: "Normal," "COVID-19," and "Viral Pneumonia."

2. **Impact of MobileNet Architecture:**
   The utilization of the MobileNet architecture in the CNN network had a positive influence on classification accuracy. The resulting model achieved an accuracy rate of 91% in all data divisions, indicating MobileNet's proficiency in distinguishing these images, particularly in identifying COVID-19 cases.

3. **Advantages of Softmax Activation Function:**
   The application of the softmax activation function in classifying COVID-19 images using MobileNet with transfer learning yielded significant results. This activation function aided the model in providing more accurate classification probabilities for each category, effectively identifying whether chest X-ray images indicated COVID-19 cases.

## Recommendations

Based on the research outcomes, several recommendations for further development are as follows:

1. **Further Validation with Diverse and Larger Datasets:**
   Despite promising classification results, further validation with larger and more diverse datasets can enhance the model's reliability and generalization in classifying COVID-19 images from various sources and lighting conditions.

2. **Exploration of Other Activation Functions:**
   In addition to softmax, experimenting with other activation functions such as ReLU, tanh, or ELU can be conducted to evaluate performance and select a more suitable activation function for the COVID-19 image classification task.

3. **Implementation of White Balance Methods in Data Preprocessing:**
   Applying white balance methods in data preprocessing can help correct color imbalances in images, especially when they originate from various sources or different lighting conditions. This can improve image consistency and quality before feature extraction by the model.

4. **Exploration of Fine-Tuning Methods:**
   Fine-tuning involves adapting a pre-trained model to a new dataset, such as the COVID-19 image dataset used in this research. Fine-tuning can lead to better alignment with specific dataset characteristics and enhance model performance. Therefore, exploring the use of fine-tuning methods on the MobileNet model can improve accuracy and generalization in the COVID-19 image classification task.

For a comprehensive analysis of the results and more detailed information, please refer to the [Report Code Number 3_MobileNetV1_701515_Report](3_MobileNetV1_701515_Report.ipynb) , and [Report Code Number 3_MobileNetV1_801010_Report](3_MobileNetV1_801010_Report.ipynb).
