# viz_forensic_dir.py
import os
import glob
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def list_images(root: str):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

def load_and_preprocess(path: str, img_size: int, jpeg_q: int | None):
    """
    - robust path reading (supports non-ascii paths)
    - RGB conversion
    - resize to (img_size, img_size)
    - optional JPEG re-encode (jpeg_q)
    """
    img_array = np.fromfile(path, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if jpeg_q is not None:
        ok, enc = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
        )
        if not ok:
            raise ValueError("cv2.imencode failed.")
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img_final = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    else:
        img_final = img_resized

    return img_final.astype(np.uint8)

def rgb_to_gray01(img_rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


# -----------------------------
# FFT Spectrum + 1D Radial Profile
# -----------------------------
def fft_log_spectrum(gray01: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    mag = np.abs(F).astype(np.float32)
    logmag = np.log(mag + 1e-8)
    logmag = (logmag - logmag.min()) / (logmag.max() - logmag.min() + 1e-8)
    return logmag  # in [0,1]

def radial_profile(power2d: np.ndarray, n_bins: int = 64) -> np.ndarray:
    H, W = power2d.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r = r / (r.max() + 1e-8)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        out[i] = power2d[m].mean() if np.any(m) else 0.0
    return np.log(out + 1e-8)

def spectrum_1d(gray01: np.ndarray, n_bins: int = 64) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    power = (np.abs(F) ** 2).astype(np.float32)
    return radial_profile(power, n_bins=n_bins)


# -----------------------------
# Saturation Histogram
# -----------------------------
def saturation_hist(img_rgb_u8: np.ndarray, bins: int = 32):
    hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    hist, edges = np.histogram(s.reshape(-1), bins=bins, range=(0.0, 1.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist


# -----------------------------
# Residual + MSCN
# -----------------------------
def residual_map(gray01: np.ndarray, mode: str = "highpass") -> np.ndarray:
    """
    Returns a visualization map in [0,1]:
    0.5 = near-zero residual, brighter/darker = stronger residual.
    """
    if mode == "highpass":
        blur = cv2.GaussianBlur(gray01, (0, 0), sigmaX=1.0)
        res = gray01 - blur
    elif mode == "denoise":
        g_u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
        dn = cv2.fastNlMeansDenoising(g_u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
        dn01 = dn.astype(np.float32) / 255.0
        res = gray01 - dn01
    else:
        raise ValueError("residual_mode must be 'highpass' or 'denoise'")

    vmax = np.percentile(np.abs(res), 99) + 1e-8
    vis = np.clip(res / vmax * 0.5 + 0.5, 0, 1)
    return vis.astype(np.float32)

def box_filter(gray01: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    gray01 = gray01.astype(np.float32, copy=False)
    return cv2.boxFilter(
        gray01, ddepth=-1, ksize=(k, k),
        normalize=True, borderType=cv2.BORDER_REFLECT
    )

def mscn_map(gray01: np.ndarray, r: int = 3) -> np.ndarray:
    """
    MSCN visualization in [0,1]:
    values mapped from clipped MSCN z-scores [-3,3] -> [0,1]
    0.5 ~ zero, extremes show local statistical deviations.
    """
    mu = box_filter(gray01, r)
    mu2 = box_filter(gray01 * gray01, r)
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 1e-8))
    mscn = (gray01 - mu) / (sigma + 1e-8)

    v = np.clip(mscn, -3, 3)
    return ((v + 3) / 6.0).astype(np.float32)


# -----------------------------
# Wavelet (Haar) Visualization
# -----------------------------
def haar1(gray01: np.ndarray):
    """
    1-level 2D Haar decomposition.
    Returns:
      a: approximation (low-low)
      h: HL (horizontal detail)
      v: LH (vertical detail)
      d: HH (diagonal detail)
    """
    g = gray01.astype(np.float32, copy=False)
    a = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]) / 2.0
    h = (g[0::2, 0::2] - g[0::2, 1::2] + g[1::2, 0::2] - g[1::2, 1::2]) / 2.0  # HL
    v = (g[0::2, 0::2] + g[0::2, 1::2] - g[1::2, 0::2] - g[1::2, 1::2]) / 2.0  # LH
    d = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2.0  # HH
    return a, h, v, d

def band_to_vis(band: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    """
    Convert wavelet band to [0,1] for visualization using z-score clipping.
    """
    b = band.astype(np.float32)
    m = float(b.mean())
    s = float(b.std() + 1e-8)
    z = (b - m) / s
    z = np.clip(z, -clip_std, clip_std)
    vis = (z + clip_std) / (2.0 * clip_std)
    return vis.astype(np.float32)

def wavelet_maps(gray01: np.ndarray, levels: int = 2):
    """
    Returns a dict of wavelet subband visualization maps in [0,1].
    L1: H/V/D
    L2: H/V/D (computed on approximation a1)
    """
    out = {}
    a1, h1, v1, d1 = haar1(gray01)
    out["L1_H(HL)"] = band_to_vis(h1)
    out["L1_V(LH)"] = band_to_vis(v1)
    out["L1_D(HH)"] = band_to_vis(d1)

    if levels >= 2:
        a2, h2, v2, d2 = haar1(a1)
        out["L2_H(HL)"] = band_to_vis(h2)
        out["L2_V(LH)"] = band_to_vis(v2)
        out["L2_D(HH)"] = band_to_vis(d2)
    return out


# -----------------------------
# GLCM Heatmap (optional)
# -----------------------------
def glcm_matrix(img01: np.ndarray, levels: int = 16, dx: int = 1, dy: int = 0) -> np.ndarray:
    q = np.clip((img01 * (levels - 1)).astype(np.int32), 0, levels - 1)
    H, W = q.shape
    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = max(0, -dx), max(0, -dy)
    a = q[y1:H-y2, x1:W-x2]
    b = q[y2:H-y1, x2:W-x1]

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a.reshape(-1), b.reshape(-1)), 1.0)
    M = M / (M.sum() + 1e-8)
    return M


# -----------------------------
# Plot helpers
# -----------------------------
def save_grid(images, titles, out_path, ncols=4):
    n = len(images)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(4*ncols, 4*nrows))
    for i, (img, t) in enumerate(zip(images, titles), start=1):
        plt.subplot(nrows, ncols, i)
        if img.ndim == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(t, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_overlay_curves(x, y_real, y_fake, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y_real, label="REAL")
    plt.plot(x, y_fake, label="FAKE")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--jpeg_q", type=int, default=0, help="0 to disable JPEG re-encode")
    ap.add_argument("--residual_mode", choices=["highpass", "denoise"], default="highpass")
    ap.add_argument("--spec_bins", type=int, default=64)
    ap.add_argument("--sat_bins", type=int, default=32)
    ap.add_argument("--glcm_levels", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jpeg_q = args.jpeg_q if args.jpeg_q > 0 else None

    real_paths = list_images(args.real_dir)
    fake_paths = list_images(args.fake_dir)
    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError("No images found. Check directories and extensions.")

    rng = random.Random(args.seed)
    real_sel = rng.sample(real_paths, k=min(args.n_samples, len(real_paths)))
    fake_sel = rng.sample(fake_paths, k=min(args.n_samples, len(fake_paths)))

    # ---------- Sample grid (original) ----------
    real_imgs = [load_and_preprocess(p, args.img_size, jpeg_q) for p in real_sel]
    fake_imgs = [load_and_preprocess(p, args.img_size, jpeg_q) for p in fake_sel]
    save_grid(real_imgs, [f"REAL\n{os.path.basename(p)}" for p in real_sel],
              os.path.join(args.out_dir, "01_real_samples.png"))
    save_grid(fake_imgs, [f"FAKE\n{os.path.basename(p)}" for p in fake_sel],
              os.path.join(args.out_dir, "02_fake_samples.png"))

    # ---------- Spectrum + saturation (mean curves over samples) ----------
    real_spec, fake_spec = [], []
    real_sat, fake_sat = [], []

    # grids: FFT / Residual / MSCN / Wavelet
    real_fft_imgs, fake_fft_imgs = [], []
    real_res_imgs, fake_res_imgs = [], []
    real_mscn_imgs, fake_mscn_imgs = [], []

    real_wav_imgs, fake_wav_imgs = [], []
    real_wav_titles, fake_wav_titles = [], []

    # ---------- REAL loops ----------
    for img in real_imgs:
        g = rgb_to_gray01(img)

        real_fft_imgs.append(fft_log_spectrum(g))
        real_res_imgs.append(residual_map(g, mode=args.residual_mode))
        real_mscn_imgs.append(mscn_map(g, r=3))

        real_spec.append(spectrum_1d(g, n_bins=args.spec_bins))
        c, h = saturation_hist(img, bins=args.sat_bins)
        real_sat.append(h)

        wav = wavelet_maps(g, levels=2)
        for k, vmap in wav.items():
            real_wav_imgs.append(vmap)
            real_wav_titles.append(f"REAL {k}")

    # ---------- FAKE loops ----------
    for img in fake_imgs:
        g = rgb_to_gray01(img)

        fake_fft_imgs.append(fft_log_spectrum(g))
        fake_res_imgs.append(residual_map(g, mode=args.residual_mode))
        fake_mscn_imgs.append(mscn_map(g, r=3))

        fake_spec.append(spectrum_1d(g, n_bins=args.spec_bins))
        c2, h2 = saturation_hist(img, bins=args.sat_bins)
        fake_sat.append(h2)

        wav = wavelet_maps(g, levels=2)
        for k, vmap in wav.items():
            fake_wav_imgs.append(vmap)
            fake_wav_titles.append(f"FAKE {k}")

    # ---------- Mean curves ----------
    real_spec_mean = np.mean(np.stack(real_spec, axis=0), axis=0)
    fake_spec_mean = np.mean(np.stack(fake_spec, axis=0), axis=0)
    x_spec = np.arange(args.spec_bins)
    save_overlay_curves(
        x_spec, real_spec_mean, fake_spec_mean,
        "Radial bin (low → high frequency)", "log power",
        "Mean 1D Radial Spectrum (REAL vs FAKE)",
        os.path.join(args.out_dir, "03_mean_spectrum_1d.png")
    )

    real_sat_mean = np.mean(np.stack(real_sat, axis=0), axis=0)
    fake_sat_mean = np.mean(np.stack(fake_sat, axis=0), axis=0)
    save_overlay_curves(
        c, real_sat_mean, fake_sat_mean,
        "Saturation (0 → 1)", "Density",
        "Mean Saturation Histogram (REAL vs FAKE)",
        os.path.join(args.out_dir, "04_mean_saturation_hist.png")
    )

    # ---------- Grids for FFT / Residual / MSCN ----------
    save_grid(real_fft_imgs, [f"REAL FFT\n{i}" for i in range(len(real_fft_imgs))],
              os.path.join(args.out_dir, "05_real_fft_grid.png"), ncols=4)
    save_grid(fake_fft_imgs, [f"FAKE FFT\n{i}" for i in range(len(fake_fft_imgs))],
              os.path.join(args.out_dir, "06_fake_fft_grid.png"), ncols=4)

    save_grid(real_res_imgs, [f"REAL residual\n{i}" for i in range(len(real_res_imgs))],
              os.path.join(args.out_dir, "07_real_residual_grid.png"), ncols=4)
    save_grid(fake_res_imgs, [f"FAKE residual\n{i}" for i in range(len(fake_res_imgs))],
              os.path.join(args.out_dir, "08_fake_residual_grid.png"), ncols=4)

    save_grid(real_mscn_imgs, [f"REAL MSCN\n{i}" for i in range(len(real_mscn_imgs))],
              os.path.join(args.out_dir, "09_real_mscn_grid.png"), ncols=4)
    save_grid(fake_mscn_imgs, [f"FAKE MSCN\n{i}" for i in range(len(fake_mscn_imgs))],
              os.path.join(args.out_dir, "10_fake_mscn_grid.png"), ncols=4)

    # ---------- GLCM example heatmaps (first sample only) ----------
    glcm_real = glcm_matrix(real_res_imgs[0], levels=args.glcm_levels, dx=1, dy=0)
    glcm_fake = glcm_matrix(fake_res_imgs[0], levels=args.glcm_levels, dx=1, dy=0)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(glcm_real, cmap="gray"); plt.title("REAL Residual-GLCM"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(glcm_fake, cmap="gray"); plt.title("FAKE Residual-GLCM"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "11_glcm_compare.png"), dpi=200)
    plt.close()

    # ---------- Wavelet grids ----------
    save_grid(real_wav_imgs, real_wav_titles,
              os.path.join(args.out_dir, "12_real_wavelet_grid.png"), ncols=3)
    save_grid(fake_wav_imgs, fake_wav_titles,
              os.path.join(args.out_dir, "13_fake_wavelet_grid.png"), ncols=3)

    print(f"Saved all visualizations to: {args.out_dir}")
    print("Files:")
    print(" 01/02 samples")
    print(" 03 spectrum curve, 04 saturation curve")
    print(" 05/06 FFT grids, 07/08 residual grids, 09/10 MSCN grids")
    print(" 11 residual-GLCM compare")
    print(" 12/13 wavelet grids")

if __name__ == "__main__":
    main()
