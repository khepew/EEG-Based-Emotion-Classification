# EEG-Based-Emotion-Classification

This repository contains the implementation of a project focused on EEG-based emotion recognition using a dataset collected in an immersive virtual reality (VR) environment. The dataset and methodology were inspired by the study titled "EEG-based emotion recognition in an immersive virtual reality environment: From local activity to brain network features" and further developed as part of a computational intelligence course project.

## Project Overview

### Objective
The main objective of this project is to classify emotions into two categories: positive and negative, based on EEG signals recorded while participants were exposed to different VR scenarios. The project involves feature extraction, selection, and classification using machine learning techniques.

### Dataset
The dataset used in this project consists of EEG recordings from 25 participants (after excluding some due to poor data quality) while they watched 3D VR videos designed to elicit positive, neutral, and negative emotions. The EEG data was recorded from 59 channels at a sampling rate of 1000 Hz, with each recording lasting 4 seconds.

- **Total Samples**: 709 (550 for training, 159 for testing)
- **EEG Channels**: 59
- **Sampling Rate**: 1000 Hz
- **Labels**: Positive emotion (1) and Negative emotion (-1)

### Features
For each EEG channel, the following features were extracted:
- Variance
- Form Factor
- Mean Frequency
- Median Frequency
- Coefficients of Autoregressive Model
- Occupied Bandwidth
- Band Power
- Maximum Power Frequency
- Skewness
- Average Power
- Kurtosis
- Standard Deviation (STD)
- Crest Factor
- Zero Crossing Rate
- Spectral Density
- Dominant Frequency

### Feature Selection
Fisher scores were calculated for each feature to select the most relevant ones. A threshold of 0.8 was used to find correlated channels, and the top 250 features were selected based on Fisher scores.

### Classification Methods
Two neural network models were implemented for classification:
1. **Multi-Layer Perceptron (MLP)**
   - Activation Functions: `transig`, `hardlims`, and `purelin`
   - The best performing model used the `purelin` activation function with 20 hidden neurons.

2. **Radial Basis Function (RBF) Network**
   - The RBF network was also trained using the selected features.

### Results
The MLP model with the `purelin` activation function showed the best performance in classifying the test data. The results of the MLP and RBF networks are saved in `test_predictions_MLP1.mat` and `test_predictions_RBF1.mat`, respectively.

## References
- Yu, M., Xiao, S., Hua, M., Wang, H., Chen, X., Tian, F., & Li, Y. (2022). EEG-based emotion recognition in an immersive virtual reality environment: From local activity to brain network features. *Biomedical Signal Processing and Control*, 72, 103349.
