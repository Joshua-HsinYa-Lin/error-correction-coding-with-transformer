import itertools
import h5py
import numpy as np

def length_profile(file_path: str, num_samples: int = 5000) -> None:
    original_lengths = []
    edited_lengths = []
    length_deltas = []
    equal_length_subs = []
    with h5py.File(file_path, "r") as f:
        for key in itertools.islice(f.keys(), num_samples):
            entry = f[key]
            original = np.array(entry["original"])
            edited = np.array(entry["edited"])
            original_lengths.append(len(original))
            edited_lengths.append(len(edited))
            length_deltas.append(len(edited) - len(original))
            if len(original) == len(edited):
                equal_length_subs.append(int((original != edited).sum()))
    original_lengths = np.array(original_lengths)
    edited_lengths = np.array(edited_lengths)
    length_deltas = np.array(length_deltas)
    print(f"Samples: {len(original_lengths)}")
    print(f"Original length min/max/std: {original_lengths.min()} {original_lengths.max()} {original_lengths.std():.3f}")
    print(f"Edited length min/max/std: {edited_lengths.min()} {edited_lengths.max()} {edited_lengths.std():.3f}")
    print(f"Fraction with equal length: {np.mean(length_deltas == 0):.4f}")
    print(f"Length delta min/max: {length_deltas.min()} {length_deltas.max()}")
    if equal_length_subs:
        equal_length_subs = np.array(equal_length_subs)
        print(f"Among equal-length pairs, substitutions per sequence mean/std: {equal_length_subs.mean():.3f} {equal_length_subs.std():.3f}")
        print(f"Implied substitution rate: {equal_length_subs.mean() / original_lengths.mean():.4f}")

if __name__ == "__main__":
    length_profile("dataset/edit_distance_dataset_K_90.h5", num_samples=5000)
