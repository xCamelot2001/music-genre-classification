# Music Genre Classification - COMP6252 Coursework

This repository contains improved implementations of neural network architectures for classifying music genres using the GTZAN dataset, as required for COMP6252 Coursework 1.

## Overview

The project implements six different neural network architectures for music genre classification:

1. **Net1**: A fully connected network with two hidden layers
2. **Net2**: A convolutional network with custom parameters
3. **Net3**: A convolutional network with batch normalization
4. **Net4**: Same as Net3 but with RMSProp optimizer
5. **Net5**: An RNN network with LSTMs (already implemented and working well)
6. **Net6**: Same as Net5 with GANs for data augmentation (already implemented and working well)

For models 1-4, the code has been improved with:
- Better regularization techniques (dropout, batch normalization)
- Enhanced data augmentation
- Learning rate scheduling
- Improved training loops with early stopping
- Comprehensive analysis utilities

## Improvements Made

### Net1 - Fully Connected Network
- Added proper batch normalization for better training stability
- Added dropout layers to prevent overfitting
- Implemented better weight initialization
- Added adaptive learning rate scheduler

### Net2 - Convolutional Network
- Improved channel dimensions for better feature extraction
- Added dropout layers to prevent overfitting
- Optimized kernel sizes and stride values
- Implemented proper learning rate scheduling

### Net3 - CNN with Batch Normalization
- Added batch normalization after each convolutional layer
- Implemented dropout in both convolutional and fully connected layers
- Increased hidden layer dimensions
- Added learning rate scheduling

### Net4 - CNN with Batch Normalization & RMSProp
- Same architecture as Net3 but uses RMSProp optimizer
- Tuned RMSProp hyperparameters for better performance
- Added weight decay for better generalization

### Data Augmentation
- Enhanced image augmentation techniques:
  - Random resized crops
  - Random rotations
  - Random horizontal flips
  - Random affine transformations
  - Random erasing
  - Gaussian noise

### Training and Evaluation
- Improved training loop with better metrics tracking
- Implemented early stopping to prevent overfitting
- Added learning rate scheduling for optimal convergence
- Comprehensive model evaluation and analysis tools

## Project Structure

- **net1_improved.py**: Improved implementation of Net1
- **net2_improved.py**: Improved implementation of Net2
- **net3_improved.py**: Improved implementation of Net3
- **net4_improved.py**: Improved implementation of Net4
- **data_augmentation.py**: Enhanced data loading and augmentation
- **modified_train_function.py**: Improved training and evaluation functions
- **model_analysis.py**: Utilities for analyzing model performance
- **evaluate_models.py**: Script to evaluate trained models
- **report_helper.py**: Utilities for generating report data
- **main_script.py**: Main execution script

## How to Run

1. **Setup Environment**:
   ```bash
   pip install torch torchvision matplotlib numpy pandas scikit-learn seaborn librosa
   ```

2. **Prepare Data**:
   - Download the GTZAN dataset from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
   - Extract to `./Data` directory

3. **Train Models**:
   ```bash
   python main_script.py
   ```
   This will train all four improved models for both 50 and 100 epochs each and save the results.

4. **Evaluate Models**:
   ```bash
   python evaluate_models.py
   ```
   This will evaluate all trained models and generate analysis reports and visualizations.

5. **Generate Report Data**:
   ```bash
   python report_helper.py
   ```
   This will generate additional charts and tables for the report.

## Results

After running the evaluation scripts, the following results will be available in the `results` directory:

- **Confusion matrices** for each model
- **Performance comparison** across all models
- **Per-class accuracy** analysis
- **Feature space visualization** using t-SNE
- **Training time vs accuracy** analysis
- **Model efficiency metrics**

## Report Guidelines

When preparing your report for submission, focus on:

1. **Implementation details**: Describe the key improvements made to each architecture
2. **Parameter tuning**: Explain the choices of hyperparameters and their effect on performance
3. **Results comparison**: Compare the performance of all models and discuss the findings
4. **Discussion**: Analyze why certain models perform better than others

Remember to keep your report within the 3-page limit using the provided CVPR format.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size
- If training is too slow, you can use the pre-trained models provided in the repository
- For any issues running the code, check the console output for error messages
