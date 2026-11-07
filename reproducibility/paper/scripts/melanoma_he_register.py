"""Register the DBiT brightfield (same section as RNA+ATAC) onto the ADJACENT H&E (Fig 3a).

Recovered from the interactive session of 2026-09-27 and assembled verbatim into one script;
logic, parameters and order are unchanged.

  A  thumbnails   H&E /32 and /16 by strided reads; brightfield /8
  B  coarse       tissue masks (H&E: saturation > 40 or grey < 0.75; brightfield: tissue is
                  BRIGHT, glass dark, blurred > 0.45); exhaustive template match over window width
                  220-740 px (H&E /32), rotation 0-354 deg step 6, with / without flip
  C  ECC          Euclidean ECC on the blurred masks, initialised at the midpoint of the two best
                  coarse hits (they agree: no flip, no rotation, width 440-460)
  D  Powell       direct maximisation of mask correlation over scale, rotation and shift at
                  blur sigma 6 -> 4 -> 2 (H&E /16, brightfield /8), starting from C. The scale has a
                  single clear optimum (profile in the output); ECC alone does not move the scale.
  out             registration.json: bf_to_he_full (3x3, brightfield full-res px -> H&E full-res px)

Physical scale: DBiT pitch 40 um (20 um channels, user-confirmed) = 129 BF px -> BF 0.310 um/px;
H&E um/px = BF um/px / scale (0.137). The DBiT window is a 2 x 2 mm field.

Usage: python scripts/melanoma_he_register.py [out_dir]
"""
import json, os, sys
import numpy as np, cv2, tifffile, zarr
from PIL import Image
from scipy.optimize import minimize

M = "/mnt/HDD2B/Users/yancui/Yufan_Melanoma"
TIF, BF = f"{M}/HE_HMelanoma.tif", f"{M}/spatial/tissue_lowres_image.png"
Image.MAX_IMAGE_PIXELS = None


def thumbnails(out):
    z = zarr.open(tifffile.TiffFile(TIF).aszarr(), mode="r")
    Image.fromarray(np.asarray(z[::32, ::32])).save(f"{out}/he_thumb.png")
    Image.fromarray(np.asarray(z[::16, ::16])).save(f"{out}/he_16.png")
    bf = Image.open(BF).convert("RGB")
    bf.resize((bf.size[0] // 8, bf.size[1] // 8)).save(f"{out}/bf_thumb.png")


def he_mask(he_rgb_uint8):
    hsv = cv2.cvtColor(he_rgb_uint8, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] > 40) | (he_rgb_uint8.mean(2) / 255 < 0.75)).astype(np.float32)


def coarse(out):
    he = np.asarray(Image.open(f"{out}/he_thumb.png").convert("RGB"))
    bf = np.asarray(Image.open(f"{out}/bf_thumb.png").convert("L")).astype(float) / 255
    mh = he_mask(he)
    mb = (cv2.GaussianBlur(bf, (0, 0), 3) > 0.45).astype(np.float32)
    mhs = cv2.GaussianBlur(mh, (0, 0), 4); mbs = cv2.GaussianBlur(mb, (0, 0), 4 * mb.shape[0] / 800)
    best = []
    for flip in (False, True):
        src = mbs[:, ::-1] if flip else mbs
        for W in range(220, 760, 20):
            small = cv2.resize(src, (W, W), interpolation=cv2.INTER_AREA)
            for th in range(0, 360, 6):
                Mr = cv2.getRotationMatrix2D((W / 2, W / 2), th, 1.0)
                rot = cv2.warpAffine(small, Mr, (W, W), borderValue=0)
                if rot.shape[0] >= mhs.shape[0] or rot.shape[1] >= mhs.shape[1]: continue
                r = cv2.matchTemplate(mhs, rot, cv2.TM_CCOEFF_NORMED)
                _, v, _, loc = cv2.minMaxLoc(r)
                best.append((v, flip, W, th, loc))
    best.sort(key=lambda b: -b[0])
    return best[:8]


def ecc(out, top2):
    he = np.asarray(Image.open(f"{out}/he_thumb.png").convert("RGB"))
    bf = np.asarray(Image.open(f"{out}/bf_thumb.png").convert("L")).astype(float) / 255
    mh = he_mask(he)
    mb = (cv2.GaussianBlur(bf, (0, 0), 3) > 0.45).astype(np.float32)
    mhs = cv2.GaussianBlur(mh, (0, 0), 3); mbs = cv2.GaussianBlur(mb, (0, 0), 3)
    W = float(np.mean([t[2] for t in top2])); tx = float(np.mean([t[4][0] for t in top2])); ty = float(np.mean([t[4][1] for t in top2]))
    s = W / mb.shape[1]
    warp = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    tmpl = cv2.warpAffine(mbs, warp, (mhs.shape[1], mhs.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpAffine(np.ones_like(mbs), warp, (mhs.shape[1], mhs.shape[0]), flags=cv2.INTER_NEAREST).astype(np.uint8)
    Rr = np.eye(2, 3, dtype=np.float32)
    cc, Rr = cv2.findTransformECC(tmpl, mhs, Rr, cv2.MOTION_EUCLIDEAN, crit, valid, 5)
    Fm = np.linalg.inv(np.vstack([Rr, [0, 0, 1]])) @ np.vstack([warp, [0, 0, 1]])
    return np.diag([32.0, 32.0, 1]) @ Fm @ np.diag([1 / 8.0, 1 / 8.0, 1]), float(cc), dict(W=W, tx=tx, ty=ty)


def powell(out, F0):
    he = np.asarray(Image.open(f"{out}/he_16.png").convert("RGB"))
    bfF = np.asarray(Image.open(BF).convert("L")).astype(float) / 255
    DS = 8; bf = cv2.resize(bfF, (bfF.shape[1] // DS, bfF.shape[0] // DS), interpolation=cv2.INTER_AREA)
    mh = he_mask(he)
    mb = (cv2.GaussianBlur(bf, (0, 0), 2) > 0.45).astype(np.float32)
    s0 = np.sqrt(abs(np.linalg.det(F0[:2, :2]))); th0 = np.arctan2(F0[1, 0], F0[0, 0])
    cx = cy = 6382 / 2
    c_he0 = (F0 @ [cx, cy, 1])[:2]

    def Fmat(p):
        s, th, dx, dy = p
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * s
        c = c_he0 + [dx, dy]
        F = np.eye(3); F[:2, :2] = Rm; F[:2, 2] = c - Rm @ [cx, cy]; return F
    small = lambda F: np.diag([1 / 16, 1 / 16, 1]) @ F @ np.diag([DS, DS, 1])
    cache = {}

    def corr(p, sig):
        Fs = small(Fmat(p))[:2].astype(np.float32)
        if sig not in cache: cache[sig] = (cv2.GaussianBlur(mb, (0, 0), sig * 16 / DS / 2.26 / (1)), cv2.GaussianBlur(mh, (0, 0), sig))
        B, Hm = cache[sig]
        w = cv2.warpAffine(B, Fs, (he.shape[1], he.shape[0]), borderValue=-1)
        v = w > -0.5
        return np.corrcoef(w[v], Hm[v])[0, 1]
    p0 = np.array([s0, th0, 0, 0]); p = p0.copy()
    for sig in (6, 4, 2):
        obj = lambda q: -corr(np.array([q[0], q[1], q[2] * 100, q[3] * 100]), sig)
        r = minimize(obj, [p[0], p[1], p[2] / 100, p[3] / 100], method="Powell",
                     options=dict(xtol=1e-4, ftol=1e-5, maxfev=3000))
        p = np.array([r.x[0], r.x[1], r.x[2] * 100, r.x[3] * 100])
    prof = []
    for sc in np.linspace(p[0] * 0.8, p[0] * 1.2, 17):
        obj = lambda q: -corr(np.array([sc, q[0], q[1] * 100, q[2] * 100]), 2)
        r = minimize(obj, [p[1], p[2] / 100, p[3] / 100], method="Powell", options=dict(xtol=1e-4, ftol=1e-5, maxfev=1500))
        prof.append((float(sc), float(-r.fun)))
    return Fmat(p), dict(scale=float(p[0]), rot_deg=float(np.degrees(p[1])), shift=[float(p[2]), float(p[3])],
                         corr=float(corr(p, 2)), corr_start=float(corr(p0, 2)), scale_profile=prof)


def main(out):
    os.makedirs(out, exist_ok=True)
    thumbnails(out)
    top = coarse(out); print("coarse top hits:", [(round(v, 3), f, W, t, l) for v, f, W, t, l in top[:4]], flush=True)
    assert not top[0][1] and not top[1][1], "best coarse hits should be unflipped"
    F_ecc, cc, init = ecc(out, top[:2]); print(f"ECC corr {cc:.4f} from init {init}", flush=True)
    F, info = powell(out, F_ecc)
    print(f"final: scale {info['scale']:.4f} HE px / BF px, rotation {info['rot_deg']:.2f} deg, corr {info['corr']:.4f}", flush=True)
    json.dump({"similarity-free-scale": dict(bf_to_he_full=F.tolist(), scale=info["scale"], rot=info["rot_deg"], corr=info["corr"]),
               "coarse_top": [[float(v), bool(f), int(W), int(t), list(map(int, l))] for v, f, W, t, l in top],
               "ecc": dict(bf_to_he_full=F_ecc.tolist(), corr=cc, init=init), "powell": info},
              open(f"{out}/registration.json", "w"), indent=1)
    print(f"-> {out}/registration.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_istar/registration")
