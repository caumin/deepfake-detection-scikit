import numpy as np

def haar1(gray):
    # gray: (H,W) float32
    a = (gray[0::2, 0::2] + gray[0::2, 1::2] + gray[1::2, 0::2] + gray[1::2, 1::2]) / 2.0
    h = (gray[0::2, 0::2] - gray[0::2, 1::2] + gray[1::2, 0::2] - gray[1::2, 1::2]) / 2.0  # HL
    v = (gray[0::2, 0::2] + gray[0::2, 1::2] - gray[1::2, 0::2] - gray[1::2, 1::2]) / 2.0  # LH
    d = (gray[0::2, 0::2] - gray[0::2, 1::2] - gray[1::2, 0::2] + gray[1::2, 1::2]) / 2.0  # HH
    return a, h, v, d

def moments(x):
    x = x.reshape(-1)
    m = x.mean()
    s = x.std() + 1e-8
    z = x - m
    kurt = np.mean(z**4) / (s**4)
    return float(m), float(s), float(kurt)

def wavelet_features(gray, levels=2, prefix="wav"):
    feats = {}
    cur = gray.astype(np.float32)
    for lv in range(1, levels+1):
        a, h, v, d = haar1(cur)
        for name, band in [("h",h),("v",v),("d",d)]:
            m, s, k = moments(band)
            feats[f"{prefix}{lv}_{name}_mean"] = m
            feats[f"{prefix}{lv}_{name}_std"] = s
            feats[f"{prefix}{lv}_{name}_kurt"] = k
        # energy ratios
        eh = np.mean(np.abs(h)); ev = np.mean(np.abs(v)); ed = np.mean(np.abs(d))
        feats[f"{prefix}{lv}_ed_over_sum"] = float(ed / (eh+ev+ed+1e-8))
        cur = a
    return feats

def phase_features(gray, bins=16, prefix="phase"):
    F = np.fft.fft2(gray)
    ph = np.angle(F)  # [-pi, pi]
    # hist
    hist, _ = np.histogram(ph.reshape(-1), bins=bins, range=(-np.pi, np.pi), density=True)
    feats = {f"{prefix}_hist_{i:02d}": float(hist[i]) for i in range(bins)}
    # phase gradient stats
    dx = np.diff(ph, axis=1); dy = np.diff(ph, axis=0)
    feats[f"{prefix}_grad_mean"] = float((np.abs(dx).mean() + np.abs(dy).mean())/2)
    feats[f"{prefix}_grad_std"]  = float((np.abs(dx).std()  + np.abs(dy).std())/2)
    return feats

def nlf_features(gray, residual, n_bins=8, prefix="nlf"):
    # gray in [0,1], residual same shape
    g = gray.reshape(-1)
    r = residual.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins+1)
    vars_ = []
    for i in range(n_bins):
        m = (g >= edges[i]) & (g < edges[i+1])
        if m.sum() < 50:
            vars_.append(0.0); continue
        vars_.append(float(np.var(r[m])))
    vars_ = np.array(vars_, dtype=np.float32)
    feats = {f"{prefix}_var_{i:02d}": float(vars_[i]) for i in range(n_bins)}
    # slope (simple linear fit)
    x = (edges[:-1] + edges[1:]) / 2.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, vars_, rcond=None)[0]
    feats[f"{prefix}_slope"] = float(slope)
    feats[f"{prefix}_intercept"] = float(intercept)
    feats[f"{prefix}_var_std"] = float(vars_.std())
    return feats
