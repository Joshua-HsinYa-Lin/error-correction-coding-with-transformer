import torch

def ctc_greedy_decode(logits: torch.Tensor, blank: int = 0) -> list:
    preds = logits.argmax(dim=-1)
    decoded = []
    for seq in preds.tolist():
        prev = blank
        out = []
        for p in seq:
            if p != prev and p != blank:
                out.append(p)
            prev = p
        decoded.append(out)
    return decoded

def alignment_counts(pred: list, true: list) -> dict:
    n = len(pred)
    m = len(true)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == true[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i = n
    j = m
    matches = 0
    substitutions = 0
    insertions = 0
    deletions = 0
    while i > 0 or j > 0:
        diagonal = i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if pred[i - 1] == true[j - 1] else 1)
        if diagonal:
            if pred[i - 1] == true[j - 1]:
                matches += 1
            else:
                substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    distance = substitutions + insertions + deletions
    return {"matches": matches, "substitutions": substitutions, "insertions": insertions, "deletions": deletions, "distance": distance}

def sample_metrics(pred_bits: list, true_bits: list) -> dict:
    counts = alignment_counts(pred_bits, true_bits)
    distance = counts["distance"]
    aligned_length = counts["matches"] + distance
    true_length = len(true_bits)
    bit_error_rate = distance / true_length if true_length > 0 else 0.0
    hamming_loss = distance / aligned_length if aligned_length > 0 else 0.0
    block_error = 1.0 if distance > 0 else 0.0
    return {"bit_error_rate": bit_error_rate, "hamming_loss": hamming_loss, "block_error": block_error, "edit_distance": distance}

def _self_test() -> None:
    classes = [2, 0, 1, 1, 0, 1, 0, 2]
    perfect = torch.zeros(1, len(classes), 3)
    for t, c in enumerate(classes):
        perfect[0, t, c] = 5.0
    decoded = ctc_greedy_decode(perfect, blank=0)
    print(f"decode collapse/blank test: {decoded[0]} expected [2, 1, 1, 2]")
    pred_bits = [c - 1 for c in decoded[0]]
    true_bits = [1, 0, 0, 1]
    exact = sample_metrics(pred_bits, true_bits)
    print(f"exact match metrics: {exact} expected all zero")
    sub = sample_metrics([1, 0, 1, 1], [1, 0, 0, 1])
    print(f"one substitution: {sub} expected distance 1, BER 0.25")
    indel = sample_metrics([1, 0, 1], [1, 0, 0, 1])
    print(f"one deletion vs target: {indel} expected distance 1, BER 0.25")

if __name__ == "__main__":
    _self_test()
