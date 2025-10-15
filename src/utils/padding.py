import torch.nn.functional as F

def _pad_1d(x, left: int, right: int, mode: str):
    """
    x: (B, C, T). Pad trái/phải theo mode.
    - 'reflect' yêu cầu T>=2 và mỗi cạnh pad <= T-1. Nếu không thoả, fallback 'replicate'.
    """
    if left < 0 or right < 0:
        raise ValueError("left/right padding must be >= 0")

    if left == 0 and right == 0:
        return x

    if mode == "zeros":
        return F.pad(x, (left, right))

    if mode == "reflect":
        T = x.size(-1)
        if T >= 2 and left <= T-1 and right <= T-1:
            return F.pad(x, (left, right), mode="reflect")
        # fallback an toàn
        return F.pad(x, (left, right), mode="replicate")

    if mode == "replicate":
        return F.pad(x, (left, right), mode="replicate")

    raise ValueError(f"Unknown pad_mode: {mode}")


def _same_padding_lr(k: int, d: int):
    """Tính (left, right) để giữ nguyên T với conv1d (stride=1, dilation=d)."""
    P = (k - 1) * d
    left = P // 2
    right = P - left  # với k chẵn, right = left+1
    return left, right