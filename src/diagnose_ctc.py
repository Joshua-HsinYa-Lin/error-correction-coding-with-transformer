import itertools
import h5py
import numpy as np
from ctc_decode import sample_metrics

def identity_baseline(file_path: str, num_samples: int = 5000) -> None:
    bit_error_rates = []
    block_errors = []
    hamming_values = []
    edit_distances = []
    original_lengths = []
    edited_lengths = []
    with h5py.File(file_path, "r") as f:
        for key in itertools.islice(f.keys(), num_samples):
            entry = f[key]
            original = np.array(entry["original"]).tolist()
            edited = np.array(entry["edited"]).tolist()
            metrics = sample_metrics(edited, original)
            bit_error_rates.append(metrics["bit_error_rate"])
            block_errors.append(metrics["block_error"])
            hamming_values.append(metrics["hamming_loss"])
            edit_distances.append(metrics["edit_distance"])
            original_lengths.append(len(original))
            edited_lengths.append(len(edited))
    print(f"Identity baseline over {len(bit_error_rates)} samples (prediction = edited input)")
    print(f"Bit Error Rate: {np.mean(bit_error_rates):.4f}")
    print(f"Block Error Rate: {np.mean(block_errors):.4f}")
    print(f"Hamming Loss: {np.mean(hamming_values):.4f}")
    print(f"Mean Edit Distance: {np.mean(edit_distances):.4f}")
    print(f"Mean original length: {np.mean(original_lengths):.2f}")
    print(f"Mean edited length: {np.mean(edited_lengths):.2f}")

if __name__ == "__main__":
    identity_baseline("dataset/edit_distance_dataset_K_90.h5", num_samples=5000)
