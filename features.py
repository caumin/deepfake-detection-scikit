# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import fft2, fftshift

# ------------------------------------------------------------------------------
# (0) Utils
# ------------------------------------------------------------------------------

def to_gray_normalized(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float [0,1] to grayscale float [0,1]."""
    if img_rgb.ndim == 2:
        return img_rgb
    if img_rgb.shape[2] == 1:
        return img_rgb.reshape(img_rgb.shape[:2])
    
    rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray = np.dot(img_rgb, rgb_weights)
    return gray

def basic_stats(v: np.ndarray, prefix: str) -> dict:
    """Computes basic distribution statistics for a flattened array."""
    v = v.flatten()
    mean = float(v.mean())
    std = float(v.std())
    skewness = float(np.mean(((v - mean) / (std + 1e-8))**3))
    kurt = float(np.mean(((v - mean) / (std + 1e-8))**4))
    
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_skew": skewness,
        f"{prefix}_kurt": kurt,
        f"{prefix}_p01": float(np.quantile(v, 0.01)),
        f"{prefix}_p50": float(np.quantile(v, 0.50)),
        f"{prefix}_p99": float(np.quantile(v, 0.99)),
    }

def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Fast box filter implementation using OpenCV."""
    # img: (H,W) float32
    k_size = 2 * r + 1
    # Use OpenCV's boxFilter, which is fast and handles borders correctly.
    # BORDER_REFLECT is equivalent to numpy's 'reflect' mode.
    return cv2.boxFilter(img, -1, (k_size, k_size), normalize=True, borderType=cv2.BORDER_REFLECT)

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

# ------------------------------------------------------------------------------
# (0-1) DCT Utilities
# ------------------------------------------------------------------------------
def dct_matrix(N: int = 8) -> np.ndarray:
    C = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
            C[k, n] = alpha * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return C

_C8 = dct_matrix(8)

# ------------------------------------------------------------------------------
# (A) 1D Power Spectrum (Azimuthal Average) & (B) Spectral Distortion
# ------------------------------------------------------------------------------
def spectrum_features(gray_img: np.ndarray, n_bins: int) -> dict:
    """
    Computes 1D Power Spectrum and Spectral Distortion features.
    (A) 1D Power Spectrum + Azimuthal Integration
    (B) Spectral Distortion band statistics
    """
    if gray_img is None:
        return {}

    # 1. Calculate 1D Power Spectrum (Azimuthal Average)
    f = fft2(gray_img)
    fshift = fftshift(f)
    power_spectrum_2d = np.abs(fshift)**2
    
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    radii = np.sqrt((np.arange(h)[:, None] - cy)**2 + (np.arange(w)[None, :] - cx)**2)
    
    # Efficiently bin and average
    bins = np.linspace(0, min(cy, cx), n_bins + 1)
    binned_power = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (radii >= bins[i]) & (radii < bins[i+1])
        if np.any(mask):
            binned_power[i] = power_spectrum_2d[mask].mean()
            
    # Log-compress for stability
    power_spectrum_1d = np.log(binned_power + 1e-8)
    
    feats = {f"spec_{i:03d}": val for i, val in enumerate(power_spectrum_1d)}

    # 2. Calculate Spectral Distortion Features
    low_band_end = n_bins // 8
    mid_band_end = n_bins // 3
    
    E_low = np.mean(power_spectrum_1d[:low_band_end])
    E_mid = np.mean(power_spectrum_1d[low_band_end:mid_band_end])
    E_high = np.mean(power_spectrum_1d[mid_band_end:])
    
    eps = 1e-8
    feats.update({
        "spec_E_low": float(E_low),
        "spec_E_mid": float(E_mid),
        "spec_E_high": float(E_high),
        "spec_mid_over_low": float(E_mid / (E_low + eps)),
        "spec_high_over_low": float(E_high / (E_low + eps)),
        "spec_mid_over_high": float(E_mid / (E_high + eps)),
    })
    
    return feats

# ------------------------------------------------------------------------------
# (C) Color Cues
# ------------------------------------------------------------------------------
def color_features(img_rgb: np.ndarray, hist_bins: int) -> dict:
    """
    Computes color-based features: saturation stats and channel correlations.
    """
    if img_rgb is None:
        return {}
    
    feats = {}
    
    # 1. Saturation statistics from HSV
    # Use cv2 for color conversion as it's robust
    img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1] / 255.0  # Normalize to [0,1]
    
    feats.update(basic_stats(s_channel, prefix="sat"))
    
    s_hist, _ = np.histogram(s_channel.flatten(), bins=hist_bins, range=[0, 1], density=True)
    feats.update({f"sat_hist_{i:02d}": val for i, val in enumerate(s_hist)})

    # 2. RGB Channel stats and correlations
    R, G, B = img_rgb[:,:,0].ravel(), img_rgb[:,:,1].ravel(), img_rgb[:,:,2].ravel()
    feats.update(basic_stats(R, prefix="ch_r"))
    feats.update(basic_stats(G, prefix="ch_g"))
    feats.update(basic_stats(B, prefix="ch_b"))

    # Correlation matrix is more efficient
    corr_matrix = np.corrcoef([R, G, B])
    feats['corr_rg'] = float(corr_matrix[0, 1])
    feats['corr_rb'] = float(corr_matrix[0, 2])
    feats['corr_gb'] = float(corr_matrix[1, 2])
    
    return feats

# ------------------------------------------------------------------------------
# (D) Noise Residual Features
# ------------------------------------------------------------------------------
def residual_features(img_rgb: np.ndarray, mode: str, spec_bins: int) -> dict:
    """
    Computes features from noise residuals.
    - Denoise-based residual (mode='denoise')
    - High-pass residual (mode='highpass')
    """
    if img_rgb is None:
        return {}

    img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if mode == 'denoise':
        denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB) / 255.0
        residual = img_rgb - denoised_rgb
    elif mode == 'highpass':
        # Laplacian is a 2nd order derivative, good for high-pass
        residual = cv2.Laplacian((img_rgb * 255).astype(np.uint8), cv2.CV_32F) / 255.0
    else:
        raise ValueError(f"Unknown residual mode: {mode}")

    feats = {}
    # Per-channel stats
    for i, name in enumerate(['r', 'g', 'b']):
        feats.update(basic_stats(residual[:,:,i], prefix=f"res_{name}"))
        
    # Covariance of residuals
    res_r, res_g, res_b = residual[:,:,0].ravel(), residual[:,:,1].ravel(), residual[:,:,2].ravel()
    cov_matrix = np.cov([res_r, res_g, res_b])
    feats['res_cov_rg'] = float(cov_matrix[0, 1])
    feats['res_cov_rb'] = float(cov_matrix[0, 2])
    feats['res_cov_gb'] = float(cov_matrix[1, 2])
    
    # Spectrum of grayscale residual
    gray_residual = to_gray_normalized(residual)
    feats.update(spectrum_features(gray_residual, n_bins=spec_bins))
    
    return feats, gray_residual

# ------------------------------------------------------------------------------
# (E) GLCM (Gray-Level Co-occurrence Matrix) Features
# ------------------------------------------------------------------------------
def glcm(gray_img: np.ndarray, levels: int = 32, dx: int = 1, dy: int = 0) -> np.ndarray:
    """Computes the Gray-Level Co-occurrence Matrix."""
    g = np.clip((gray_img * (levels - 1)), 0, levels - 1).astype(np.int32)
    H, W = g.shape
    
    # Slices to get pixel pairs
    y_from, y_to = max(0, -dy), H - max(0, dy)
    x_from, x_to = max(0, -dx), W - max(0, dx)
    
    p1 = g[y_from:y_to, x_from:x_to]
    p2 = g[y_from+dy : y_to+dy, x_from+dx : x_to+dx]
    
    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (p1.ravel(), p2.ravel()), 1.0)
    
    # Symmetric GLCM
    M = M + M.T
    return M / (M.sum() + 1e-8)

def haralick_from_glcm(P: np.ndarray, prefix: str) -> dict:
    """Calculates Haralick features from a GLCM."""
    levels = P.shape[0]
    i, j = np.ogrid[:levels, :levels]

    with np.errstate(divide='ignore', invalid='ignore'):
        # Marginal probabilities
        px = P.sum(axis=1)
        py = P.sum(axis=0)
        
        # Means and stds of marginals
        mu_x = (i.ravel() * px).sum()
        mu_y = (j.ravel() * py).sum()
        std_x = np.sqrt(((i.ravel() - mu_x)**2 * px).sum())
        std_y = np.sqrt(((j.ravel() - mu_y)**2 * py).sum())

        contrast = ((i - j)**2 * P).sum()
        dissimilarity = (np.abs(i - j) * P).sum()
        homogeneity = (P / (1 + (i - j)**2)).sum()
        energy = (P**2).sum()
        correlation = ((i*j*P).sum() - mu_x*mu_y) / (std_x*std_y + 1e-8)
        
    return {
        f"{prefix}_contrast": float(contrast),
        f"{prefix}_dissimilarity": float(dissimilarity),
        f"{prefix}_homogeneity": float(homogeneity),
        f"{prefix}_ASM": float(energy), # Angular Second Moment
        f"{prefix}_energy": float(np.sqrt(energy)),
        f"{prefix}_correlation": float(np.nan_to_num(correlation)),
    }

def glcm_features(gray_img: np.ndarray, levels: int = 32) -> dict:
    """Computes Haralick features for multiple GLCM directions."""
    feats = {}
    directions = {"0": (1, 0), "45": (1, -1), "90": (0, -1), "135": (-1, -1)}
    
    for name, (dx, dy) in directions.items():
        P = glcm(gray_img, levels=levels, dx=dx, dy=dy)
        feats.update(haralick_from_glcm(P, prefix=f"glcm_{name}"))
        
    return feats

# ------------------------------------------------------------------------------
# (F) MSCN Features
# ------------------------------------------------------------------------------
def ms_cn_features(gray: np.ndarray, r: int = 3, prefix="mscn") -> dict:
    # gray in [0,1]
    mu = box_filter(gray, r)
    mu2 = box_filter(gray*gray, r)
    sigma = np.sqrt(np.maximum(mu2 - mu*mu, 1e-8))
    mscn = (gray - mu) / (sigma + 1e-8)

    feats = {}
    v = mscn.reshape(-1)
    # moments / percentiles
    feats.update({
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_std": float(v.std() + 1e-8),
        f"{prefix}_p01": float(np.quantile(v, 0.01)),
        f"{prefix}_p10": float(np.quantile(v, 0.10)),
        f"{prefix}_p50": float(np.quantile(v, 0.50)),
        f"{prefix}_p90": float(np.quantile(v, 0.90)),
        f"{prefix}_p99": float(np.quantile(v, 0.99)),
    })

    # neighbor products (H, V, D1, D2)
    def prod_stats(a):
        t = a.reshape(-1)
        return {
            "mean": float(t.mean()),
            "std": float(t.std() + 1e-8),
            "p10": float(np.quantile(t, 0.10)),
            "p50": float(np.quantile(t, 0.50)),
            "p90": float(np.quantile(t, 0.90)),
        }

    H = mscn[:, 1:] * mscn[:, :-1]
    V = mscn[1:, :] * mscn[:-1, :]
    D1 = mscn[1:, 1:] * mscn[:-1, :-1]
    D2 = mscn[1:, :-1] * mscn[:-1, 1:]

    for name, arr in [("H", H), ("V", V), ("D1", D1), ("D2", D2)]:
        s = prod_stats(arr)
        for k, val in s.items():
            feats[f"{prefix}_prod_{name}_{k}"] = val
    return feats

# ------------------------------------------------------------------------------
# (G) Block DCT Features
# ------------------------------------------------------------------------------
def block_dct_features(gray: np.ndarray, block: int = 8, prefix="bdct") -> dict:
    # gray in [0,1], assume 224x224 divisible by 8
    H, W = gray.shape
    C = _C8
    # collect a few aggregate stats over all blocks
    dc_vals = []
    lf_energy = []
    hf_energy = []

    # define low-frequency mask (e.g., top-left 3x3)
    lf_mask = np.zeros((block, block), dtype=bool)
    lf_mask[:3, :3] = True
    hf_mask = ~lf_mask

    for y in range(0, H, block):
        for x in range(0, W, block):
            if y+block > H or x+block > W: continue
            b = gray[y:y+block, x:x+block].astype(np.float32)
            # DCT: C * b * C^T
            d = C @ b @ C.T
            dc_vals.append(d[0,0])
            lf_energy.append(float(np.mean(np.abs(d[lf_mask]))))
            hf_energy.append(float(np.mean(np.abs(d[hf_mask]))))

    dc_vals = np.array(dc_vals, dtype=np.float32)
    lf_energy = np.array(lf_energy, dtype=np.float32)
    hf_energy = np.array(hf_energy, dtype=np.float32)

    feats = {
        f"{prefix}_dc_mean": float(dc_vals.mean()),
        f"{prefix}_dc_std": float(dc_vals.std() + 1e-8),
        f"{prefix}_lf_mean": float(lf_energy.mean()),
        f"{prefix}_lf_std": float(lf_energy.std() + 1e-8),
        f"{prefix}_hf_mean": float(hf_energy.mean()),
        f"{prefix}_hf_std": float(hf_energy.std() + 1e-8),
        f"{prefix}_hf_over_lf": float((hf_energy.mean()+1e-8)/(lf_energy.mean()+1e-8)),
    }
    return feats

# ------------------------------------------------------------------------------
# (H) Local Patch Spectrum Features
# ------------------------------------------------------------------------------
def patch_spectrum_features(gray: np.ndarray, patch: int = 32, prefix="pspec") -> dict:
    H, W = gray.shape
    # simple band ratio from FFT power (no radial bins to keep dim small)
    def band_ratio(pwr):
        # normalized radius r in [0,1]
        h, w = pwr.shape
        cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        r = r / (r.max() + 1e-8)
        low = pwr[(r < 0.15)].mean()
        mid = pwr[(r >= 0.15) & (r < 0.35)].mean()
        high = pwr[(r >= 0.35)].mean()
        return low, mid, high

    lows, mids, highs = [], [], []
    for y in range(0, H, patch):
        for x in range(0, W, patch):
            g = gray[y:y+patch, x:x+patch]
            if g.shape[0] != patch or g.shape[1] != patch:
                continue
            F = np.fft.fftshift(np.fft.fft2(g))
            pwr = (np.abs(F)**2).astype(np.float32)
            low, mid, high = band_ratio(pwr)
            lows.append(low); mids.append(mid); highs.append(high)

    lows = np.array(lows, dtype=np.float32)
    mids = np.array(mids, dtype=np.float32)
    highs = np.array(highs, dtype=np.float32)

    feats = {
        f"{prefix}_low_mean": float(np.log(np.mean(lows)+1e-8)),
        f"{prefix}_mid_mean": float(np.log(np.mean(mids)+1e-8)),
        f"{prefix}_high_mean": float(np.log(np.mean(highs)+1e-8)),
        f"{prefix}_mid_over_low": float((np.mean(mids)+1e-8)/(np.mean(lows)+1e-8)),
        f"{prefix}_high_over_low": float((np.mean(highs)+1e-8)/(np.mean(lows)+1e-8)),
        # patch variability = "국소적 아티팩트" 탐지에 도움
        f"{prefix}_high_std": float(np.log(np.std(highs)+1e-8)),
        f"{prefix}_mid_std": float(np.log(np.std(mids)+1e-8)),
    }
    return feats

# ------------------------------------------------------------------------------
# (I) Residual GLCM Features
# ------------------------------------------------------------------------------
def residual_glcm_features(gray: np.ndarray, blur_r: int = 1, levels: int = 16, prefix="rglcm") -> dict:
    # simple blur via box filter for speed (or you can reuse your GaussianBlur)
    mu = box_filter(gray, blur_r)
    res = gray - mu  # roughly [-, +]
    # normalize residual to [0,1] for quantization
    resn = (res - res.min()) / (res.max() - res.min() + 1e-8)
    q = np.clip((resn * (levels-1)).astype(np.int32), 0, levels-1)

    def glcm_res(qimg, dx, dy): # Renamed to avoid clash with existing glcm function
        H, W = qimg.shape
        x1, y1 = max(0, dx), max(0, dy)
        x2, y2 = max(0, -dx), max(0, -dy)
        a = qimg[y1:H-y2, x1:W-x2]
        b = qimg[y2:H-y1, x2:W-x1]
        M = np.zeros((levels, levels), dtype=np.float32)
        np.add.at(M, (a.reshape(-1), b.reshape(-1)), 1.0)
        M = M / (M.sum() + 1e-8)
        return M

    feats = {}
    for name, (dx, dy) in {"0":(1,0), "90":(0,1), "45":(1,1), "135":(-1,1)}.items():
        P = glcm_res(q, dx, dy)
        haralick_feats = haralick_from_glcm(P, prefix=f"rglcm_{name}") # Use existing haralick_from_glcm
        for k, val in haralick_feats.items():
            feats[k] = val # The prefix is already in haralick_from_glcm

    return feats

# ------------------------------------------------------------------------------
# (J) Wavelet Features
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# (K) Phase Features
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# (L) Noise Level Function (NLF) Features
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Top-level Feature Extractor
# ------------------------------------------------------------------------------
def extract_all_features(img_rgb: np.ndarray, spec_bins: int, color_bins: int, residual_mode: str) -> dict:
    """
    Top-level function to extract and concatenate all feature sets.
    - Converts image to float [0,1]
    - Extracts all features and returns them in a single dictionary.
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    gray_float = to_gray_normalized(img_float)

    # Dictionary to hold all features
    all_feats = {}

    # (A) & (B) Spectrum Features
    all_feats.update(spectrum_features(gray_float, n_bins=spec_bins))
    
    # (C) Color Cues
    all_feats.update(color_features(img_float, hist_bins=color_bins))
    
    # (D) Noise Residuals (using half the bins for its spectrum)
    residual_feats, gray_residual = residual_features(img_float, mode=residual_mode, spec_bins=spec_bins // 2)
    all_feats.update(residual_feats)
    
    # (E) GLCM texture features
    all_feats.update(glcm_features(gray_float, levels=32))
    
    # (F) MSCN Features
    all_feats.update(ms_cn_features(gray_float))

    # (G) Block DCT Features
    all_feats.update(block_dct_features(gray_float))

    # (H) Local Patch Spectrum Features
    all_feats.update(patch_spectrum_features(gray_float))

    # (I) Residual GLCM Features
    all_feats.update(residual_glcm_features(gray_float))

    # (J) Wavelet Features
    all_feats.update(wavelet_features(gray_float))

    # (K) Phase Features
    all_feats.update(phase_features(gray_float))

    # (L) NLF Features
    all_feats.update(nlf_features(gray_float, gray_residual))

    return all_feats