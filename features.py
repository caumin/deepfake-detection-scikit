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
    
    return feats

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
    all_feats.update(residual_features(img_float, mode=residual_mode, spec_bins=spec_bins // 2))
    
    # (E) GLCM texture features
    all_feats.update(glcm_features(gray_float, levels=32))

    return all_feats