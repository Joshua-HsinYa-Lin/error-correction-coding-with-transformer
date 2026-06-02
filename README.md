# Transformer Error Correction Coding Pipeline

**Project Overview**
This repository contains a PyTorch machine learning pipeline designed to evaluate sequence-to-sequence Transformer architectures on Error Correction Code operations. The model is benchmarked against traditional decoding algorithms such as BCJR, utilizing an embedding dimension of 256 and multi-head attention to map discrete sequence mutations into a measurable geometric space.

**Repository Architecture**
The codebase is strictly organized into dedicated directories to separate execution logic from generated artifacts.
The src directory contains all core Python execution modules and tensor mathematics.
The dataset directory is designated for the local HDF5 sequence arrays.
The plots directory stores all generated matplotlib visualizations and scatter plots.
The docs directory contains the formal IEEE LaTeX report and project context.

**Execution Modules**
The training and evaluation scripts must be executed from the root directory referencing the src folder to maintain correct pathing.
src/dataset.py handles the in-memory data loading and dynamic boolean padding masks.
src/model.py defines the Transformer encoder, decoder, and positional encoding logic.
src/train.py executes the training loop using teacher forcing and a binary cross-entropy loss function.
src/eval.py calculates the true Bit Error Rate and Block Error Rate using strict autoregressive decoding.
src/eval_metrics.py computes the Levenshtein distance correlations natively in Python.
src/pca_analysis.py extracts the high-dimensional encoder states to map the variance.

**Performance Metrics**
The architecture was evaluated on a 12,800-sample subset to prevent data leakage and ensure valid autoregressive metrics. The baseline network achieved a Bit Error Rate of 0.3716 and a Block Error Rate of 0.9999. The Levenshtein distance evaluation yielded a Mean Absolute Error of 1.9312. The distance mapping achieved a Pearson Correlation of 0.8589 and a Spearman Rank Correlation of 0.8557. Principal Component Analysis captured 86.3 percent of the total variance within a two-dimensional projection, confirming the network learned the mathematical manifold of edit operations.

**Usage Commands**
To execute the training pipeline and generate the model weights, run: python src/train.py
To generate the baseline error rates and bar charts, run: python src/eval.py
To calculate the distance correlation metrics and scatter plots, run: python src/eval_metrics.py
To map the dimensionality reduction and clustering, run: python src/pca_analysis.py