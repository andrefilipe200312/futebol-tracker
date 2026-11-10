"""Calibração inicial a partir de grupos de píxeis brancos rotulados (círculo central, linha do meio,
lateral próxima). Estima a homografia e a largura do campo. Uso: python anchor_fit.py <frame.jpg> <dets.jsonl>"""
import sys, json, cv2, numpy as np
from scipy.optimize import least_squares
from pitch import grass_mask, white_mask
from calib import proj

def fit(im, boxes, groups, H0, W0):
    """groups: dict nome -> lista de (x,y) em píxeis. Modelos: circle (r=9.15), halfway (x=0), near (y=-W/2), far (y=+W/2)."""
    C = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]], float)
    def unpack(p):
        H = cv2.getPerspectiveTransform(np.float32(C), np.float32(p[:8].reshape(4, 2))).astype(float); return H, p[8]
    def res(p):
        H, W = unpack(p); Hi = np.linalg.inv(H); r = []
        for name, pts in groups.items():
            pts = np.asarray(pts, float); q = proj(Hi, pts)
            e = proj(H, q + [0.3, 0]) - pts; e2 = proj(H, q + [0, 0.3]) - pts
            sc = (np.hypot(*e.T) + np.hypot(*e2.T)) / 0.6       # px por metro local
            if name == 'circle': d = np.hypot(q[:, 0], q[:, 1]) - 9.15
            elif name == 'halfway': d = q[:, 0]
            elif name == 'near': d = q[:, 1] + W / 2
            elif name == 'far': d = q[:, 1] - W / 2
            r.append(d * sc / np.sqrt(len(pts)))
        return np.concatenate(r)
    p0 = np.r_[proj(H0, C).ravel(), W0]
    r = least_squares(res, p0, loss='soft_l1', f_scale=0.1, x_scale=np.r_[np.full(8, 20.0), 5.0])
    H, W = unpack(r.x)
    rr = res(r.x); n = sum(len(v) for v in groups.values())
    rms = {k: float(np.sqrt(np.mean((res_k) ** 2))) for k, res_k in zip(groups, np.split(rr * 0 + 0, [0]))}  # placeholder
    return H, W, r
