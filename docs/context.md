# Project Context: Transformer for Edit Distance Evaluation

## 1. Project Goal
Combine neural networks with sequence editing distance calculations[cite: 3]. The objective is to build an independent PyTorch machine learning pipeline to evaluate Transformer models on sequence prediction and edit distance metrics against traditional baselines such as BCJR[cite: 2].

## 2. Core Concepts & Target Metrics
- **Edit Distance:** Minimum operations (insertions, deletions, substitutions) required to transform one sequence into another.
- **Error Rates:** Bit Error Rate (BER) and Block Error Rate (BLER) curves[cite: 2].
- **Machine Learning Metrics:** Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Pearson Correlation, Spearman Rank Correlation, Triplet Accuracy, and Hamming Loss[cite: 1].
- **Model Comparison:** Compare Transformer performance against baseline models including CGK, CNN-ED, GRU, and REBIND.
- **Analysis Techniques:** Utilize Principal Component Analysis (PCA) alongside Transformer metrics[cite: 1]. Execute model fine-tuning protocols[cite: 4].

## 3. Architecture Blueprint
- `dataset.py`: Utilize `h5py` to load `.h5` arrays. Convert data into PyTorch `TensorDataset` and `DataLoader` structures.
- `model.py`: Construct the network utilizing `torch.nn.Transformer` or HuggingFace module architectures.
- `train.py`: Implement the training loop, loss function calculations, `AdamW` optimizer setup, and `cuda` GPU deployment.
- `eval.py`: Execute inference, calculate target evaluation metrics, and plot predicted versus true edit distance scatter graphs alongside BER/BLER curves utilizing `matplotlib`.
