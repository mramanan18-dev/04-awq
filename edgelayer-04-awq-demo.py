"""
EdgeLayer Blog 4: AWQ demo on a synthetic layer.

Compares, on one 256 x 512 linear layer with correlated inputs and a few
hot input channels:
  - RTN (round-to-nearest)
  - keep 1% salient channels in FP16 (picked by activation size, by weight size)
  - AWQ (per-channel scaling s = s_X^alpha, grid search over alpha)
  - GPTQ (same loop as Blog 3)
Then runs a calibration-shift test: calibrate on domain A, evaluate on A and B.

All numbers are relative output error  ||WX - W_hat X|| / ||WX||  on held-out inputs.
Run:  python edgelayer-04-awq-demo.py
"""
import numpy as np

rng = np.random.default_rng(0)
D_OUT, D_IN, GROUP = 256, 512, 128
N_HOT = 5                      # ~1% of 512 input channels run hot
HOT_GAIN = 12.0                # how much bigger the hot channels are


# ---------------------------------------------------------------- data
K = 32                                              # low-rank factors -> correlated inputs
BASE = rng.normal(size=(D_IN, K)) / np.sqrt(K)      # structure shared by all text

def make_domain(seed, hot_idx, shift):
    """A 'domain' mixes the shared structure with its own (shift = 0: none, 1: all its own).
    Hot channels stay the same: they belong to the model, not the data."""
    r = np.random.default_rng(seed)
    own = r.normal(size=(D_IN, K)) / np.sqrt(K)
    A = np.sqrt(1 - shift) * BASE + np.sqrt(shift) * own
    gain = np.ones(D_IN)
    gain[hot_idx] = HOT_GAIN
    def sample(n):
        Z = r.normal(size=(K, n))
        X = A @ Z + 0.3 * r.normal(size=(D_IN, n))
        return gain[:, None] * X
    return sample

W = rng.normal(size=(D_OUT, D_IN)) * 0.02
hot = rng.choice(D_IN, N_HOT, replace=False)

dom_A = make_domain(1, hot, shift=0.0)              # calibration domain
dom_B = make_domain(2, hot, shift=0.5)              # half its structure is new
X_cal = dom_A(512)             # small calibration set
X_A   = dom_A(4096)            # held-out, same domain
X_B   = dom_B(4096)            # held-out, different domain


# ---------------------------------------------------------------- quantizer
def quant_params(Wg, bits):
    """Asymmetric min-max, one scale per row of this group."""
    qmax = 2 ** bits - 1
    lo = Wg.min(axis=1, keepdims=True)
    hi = Wg.max(axis=1, keepdims=True)
    s = np.maximum(hi - lo, 1e-12) / qmax
    z = np.round(-lo / s)
    return s, z, qmax

def quantize(w, s, z, qmax):
    q = np.clip(np.round(w / s) + z, 0, qmax)
    return (q - z) * s

def rtn(Wm, bits):
    Q = np.empty_like(Wm)
    for g in range(0, D_IN, GROUP):
        s, z, qmax = quant_params(Wm[:, g:g + GROUP], bits)
        Q[:, g:g + GROUP] = quantize(Wm[:, g:g + GROUP], s, z, qmax)
    return Q

def out_err(W_hat, X):
    return np.linalg.norm(W @ X - W_hat @ X) / np.linalg.norm(W @ X)


# ---------------------------------------------------------------- methods
def keep_fp16(bits, idx):
    Q = rtn(W, bits)
    Q[:, idx] = W[:, idx]      # these columns stay in FP16
    return Q

def awq(bits, X, grid=20, return_curve=False):
    s_x = np.abs(X).mean(axis=1)                    # average activation size per channel
    best, curve = None, []
    for alpha in np.linspace(0, 1, grid + 1):
        s = s_x ** alpha
        s = s / np.sqrt(s.max() * s.min())          # keep scales centred around 1
        W_hat = rtn(W * s, bits) / s                # Q(W·diag(s)) · diag(s)^-1
        e = out_err(W_hat, X)
        curve.append((alpha, e))
        if best is None or e < best[0]:
            best = (e, alpha, W_hat)
    return (best, curve) if return_curve else best

def gptq(bits, X, damp=0.01):
    Wc = W.copy()
    Q = np.zeros_like(W)
    H = 2 * X @ X.T
    H += damp * np.mean(np.diag(H)) * np.eye(D_IN)
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    for i in range(D_IN):
        if i % GROUP == 0:
            s, z, qmax = quant_params(Wc[:, i:i + GROUP], bits)
        w = Wc[:, i]
        Q[:, i] = quantize(w, s[:, 0], z[:, 0], qmax)
        err = (w - Q[:, i]) / U[i, i]
        Wc[:, i + 1:] -= np.outer(err, U[i, i + 1:])
    return Q


# ---------------------------------------------------------------- run
def main():
    results = {}
    act_rank = np.argsort(-np.abs(X_cal).mean(axis=1))
    wt_rank = np.argsort(-np.abs(W).mean(axis=0))
    n_keep = int(round(0.01 * D_IN))                # 1% = 5 channels

    print(f"Hot channels: {sorted(hot.tolist())}  | picked by activation: {sorted(act_rank[:n_keep].tolist())}")
    for bits in (4, 3):
        r = {}
        r["RTN"] = out_err(rtn(W, bits), X_A)
        r["1% FP16 (by weight)"] = out_err(keep_fp16(bits, wt_rank[:n_keep]), X_A)
        r["1% FP16 (by activation)"] = out_err(keep_fp16(bits, act_rank[:n_keep]), X_A)
        (e_cal, alpha, W_awq), curve = awq(bits, X_cal, return_curve=True)
        r["AWQ"] = out_err(W_awq, X_A)
        r["GPTQ"] = out_err(gptq(bits, X_cal), X_A)
        r["_alpha"] = alpha
        r["_curve"] = curve
        results[bits] = r
        print(f"\n{bits}-bit, g{GROUP}  (output error on held-out domain-A inputs)")
        for k, v in r.items():
            if not k.startswith("_"):
                print(f"  {k:<26} {100 * v:5.1f}%")
        print(f"  best alpha = {alpha:.2f}")

    # calibration shift: calibrate on A, test on A and B
    shift = {}
    for bits in (4, 3):
        _, _, W_awq = awq(bits, X_cal)
        W_gptq = gptq(bits, X_cal)
        W_rtn = rtn(W, bits)
        shift[bits] = {
            "RTN":  (out_err(W_rtn, X_A),  out_err(W_rtn, X_B)),
            "GPTQ": (out_err(W_gptq, X_A), out_err(W_gptq, X_B)),
            "AWQ":  (out_err(W_awq, X_A),  out_err(W_awq, X_B)),
        }
        print(f"\nCalibration shift, {bits}-bit (calibrated on A): test A -> test B")
        for k, (a, b) in shift[bits].items():
            print(f"  {k:<5} {100 * a:5.1f}% -> {100 * b:5.1f}%")
    return results, shift


if __name__ == "__main__":
    main()
