# Music Genre Classification

This repository contains implementations of neural network architectures for classifying music genres using the GTZAN dataset.

## Overview

The project implements six different neural network architectures for music genre classification:

1. **Net1**: A fully connected network with two hidden layers
2. **Net2**: A convolutional network with custom parameters
3. **Net3**: A convolutional network with batch normalization
4. **Net4**: Same as Net3 but with RMSProp optimizer
5. **Net5**: An RNN network with LSTMs
6. **Net6**: Same as Net5 with GANs for data augmentation

## How to Run

1. **Setup Environment**:

   ```bash
   pip install torch torchvision matplotlib numpy pandas scikit-learn seaborn librosa
   ```

2. **Prepare Data**:

   - Download the GTZAN dataset from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
   - Extract to `./Data` directory

3. **Train Models**:

4. **Evaluate Models**:

5. **Generate Report Data**:

## Results

After running the evaluation scripts, the following results will be available in the `results` directory:

- **Confusion matrices** for each model
- **Performance comparison** across all models
- **Per-class accuracy** analysis
- **Feature space visualization** using t-SNE
- **Training time vs accuracy** analysis
- **Model efficiency metrics**
