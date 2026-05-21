# Transformer-Based Error Correction Code (ECC) Pipeline

This workspace implements a sequence-to-sequence neural network utilizing a Transformer architecture to evaluate edit distances and execute sequence error corrections. The codebase is optimized for native Linux ext4 file systems and utilizes lazy-loading HDF5 dataset structures to eliminate memory allocation latency.

## Directory Structure
- GEMINI.md: Operational constraints and strict development protocols.
- context.md: Comprehensive architectural specifications and mathematical baselines.
- dataset.py: PyTorch Dataset wrapper enforcing lazy loading and multi-head attention padding masks.
- model.py: Sequence-to-sequence model defining positional encodings and multi-head dot product attention pathways.
- train.py: Optimization engine executing backward propagation, causal sequence masking, and gradient descent.

## Remote Computation Environment Setup
Execute the following commands sequentially on the compute node to construct the isolated python environment:

python3 -m venv venv
source venv/bin/activate
python3 -m pip install h5py scikit-learn scipy matplotlib editdistance
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Execution Verification Commands
To test the structural tracking of individual code modules, run the following standalone verification scripts:

1. Test dataset loading tensor dimensions:
python3 dataset.py

2. Verify model dimension transformations using dummy tensors:
python3 model.py

3. Run the complete training loop:
python3 train.py
