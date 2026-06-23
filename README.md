# Transformer Error Correction Coding Pipeline

**Project Overview**
This repository contains a PyTorch machine learning pipeline that evaluates Transformer architectures on Error Correction Code operations, benchmarked against traditional decoding algorithms such as BCJR. The pipeline uses an embedding dimension of 256 with multi-head attention. The objective is to recover original bit sequences from edited sequences and to quantify performance using Bit Error Rate, Block Error Rate, and edit distance correlation metrics.

**Architectural History**
The project passed through three architectural phases. The first phase used a standard autoregressive sequence-to-sequence Transformer, which failed because a single insertion or deletion shifts the predicted sequence and triggers cascading block errors under positional loss functions. The second phase pivoted to an encoder-only Siamese network trained with Triplet Margin Loss, which maps sequences into a continuous metric space but does not emit discrete bit sequences and therefore cannot produce a direct Bit Error Rate. The current phase pivots to a Connectionist Temporal Classification architecture, which recovers explicit bit sequences while bypassing the shift penalty of autoregressive decoding.

**Current Architecture**
The current model is the CTCTransformer defined in src/model.py. It is an encoder-only Transformer with a sinusoidal positional encoding, an embedding dimension of 256, 8 attention heads, 4 encoder layers, and a feedforward dimension of 1024. A Connectionist Temporal Classification head projects the encoder output onto three classes, comprising one blank symbol and two binary symbols. The encoder output is temporally upsampled by a factor of two using nearest-neighbor interpolation, expanding an input of length 100 into an alignment lattice of length 200. This upsampling satisfies the Connectionist Temporal Classification constraint that the input time dimension must accommodate the target length plus the number of adjacent repeated symbols.

**Repository Layout**
The src directory contains the Python execution modules and tensor mathematics. The dataset directory holds the local HDF5 sequence arrays. The plots directory stores generated matplotlib visualizations. The docs directory contains the LaTeX report and project context.

**Connectionist Temporal Classification Modules**
src/dataset_ctc.py loads the edit distance HDF5 file, offsets symbols by one to reserve index zero for the Connectionist Temporal Classification blank, applies boolean padding masks, and returns edited and original sequence pairs.
src/model.py defines the CTCTransformer, the positional encoding, and the temporal upsampling logic.
src/train_ctc.py executes the production training loop on the large edit distance dataset, deploying to CUDA when available and computing the Connectionist Temporal Classification loss with a fixed input length of 200.
src/test_ctc.py runs a single forward and backward pass on the CPU to verify that the loss computes and the gradient flows without triggering an infinite alignment cost.
src/dataset_ctc_triplet.py adapts the smaller triplet HDF5 file for Connectionist Temporal Classification by mapping each positive sequence to the model input and each anchor sequence to the recovery target.
src/train_ctc_verify.py runs a short CPU training loop on the triplet adapter to confirm loss convergence before committing to the large dataset on a CUDA device.
src/ctc_decode.py implements greedy Connectionist Temporal Classification decoding, which collapses repeated emissions and removes the blank symbol, together with a length-aware Levenshtein alignment that counts matches, substitutions, insertions, and deletions to compute the Bit Error Rate, Hamming Loss, and Block Error indicator under unequal sequence lengths.
src/eval_ctc.py loads the trained weights, trims each decoded sequence to its valid upsampled time region, and aggregates the Bit Error Rate, Block Error Rate, Hamming Loss, and mean reconstruction edit distance, saving the error rate chart and edit distance histogram to the plots directory.

**Legacy Siamese and Generative Modules**
src/dataset.py handles in-memory loading and dynamic boolean padding masks for the earlier phases.
src/train.py executes the earlier teacher-forcing training loop with a binary cross-entropy loss.
src/eval.py calculates the Bit Error Rate and Block Error Rate under autoregressive decoding.
src/eval_metrics.py computes Levenshtein distance correlations in Python.
src/pca_analysis.py extracts encoder states and maps the variance into a two-dimensional projection.

**Legacy Performance Metrics**
The Siamese phase was evaluated on a 12,800-sample subset to prevent data leakage. That network achieved a Triplet Accuracy of 0.7238, an RMSE of 2.092, and an MAE of 1.670 after fitting a linear projection from Euclidean embedding distance to Levenshtein distance. Principal Component Analysis captured 86.3 percent of the variance in a two-dimensional projection. These figures describe the Siamese phase and do not yet apply to the Connectionist Temporal Classification model. Bit Error Rate and Block Error Rate for the current architecture are pending convergence and evaluation.

**Usage Commands**
To verify the Connectionist Temporal Classification loss on the CPU, run: python src/test_ctc.py
To verify loss convergence on the smaller triplet dataset on the CPU, run: python src/train_ctc_verify.py
To execute the production training loop on the large dataset and a CUDA device, run: python src/train_ctc.py
To decode the trained model and report the Bit Error Rate, Block Error Rate, and Hamming Loss, run: python src/eval_ctc.py
To verify the decoder and alignment mathematics on synthetic sequences, run: python src/ctc_decode.py
The legacy Siamese pipeline is driven through python src/train.py, python src/eval.py, python src/eval_metrics.py, and python src/pca_analysis.py.
All scripts must be executed from the repository root to maintain correct relative pathing into the dataset directory.
