"""Ajuste do modelo físico da câmara a pontos marcados à mão (marcar_pontos.py) e a linhas brancas.
Posição da câmara, lente (k1) e comprimento do campo são comuns; cada frame-âncora tem a sua orientação e focal.
Pontos: erro de reprojeção em píxeis. Linhas do frame do círculo central (círculo, linha do meio, laterais): distância
no campo (m) x peso. Grava <saida>.json para calibrar_trecho.py.
Uso: python fit_camara.py <trecho.mp4> <dets.jsonl> <saida.json> <ancora1.json> [<ancora2.json> ...]"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera import look
from pitch import grass_mask, white_mask
from calibrar_trecho import Cam

video, dets, out, *ancs = sys.argv[1:]
W = 64.0; F_LINHAS = 600                                    # frame com o círculo central e a linha do meio
D = {d['f']: d for d in (json.loads(l) for l in open(dets) if l.strip())}
cap = cv2.VideoCapture(video)
def ler(f): cap.set(cv2.CAP_PROP_POS_FRAMES, f); return cap.read()[1]
A = [json.load(open(a)) for a in ancs]
def campo(a, L):                                            # pontos da área -> campo, com o comprimento L
    s = 1 if a['baliza'] == 'direita' else -1
    return np.array([[s * (L / 2 - p['campo'][0]), s * p['campo'][1]] for p in a['pontos']])
UV = [np.array([p['imagem'] for p in a['pontos']], float) for a in A]

im = ler(F_LINHAS); w = white_mask(im, grass_mask(im), D[F_LINHAS]['persons']); ys, xs = np.nonzero(w); P = np.c_[xs, ys].astype(float)
rng = np.random.default_rng(0); sub = lambda X, n: X[rng.choice(len(X), min(n, len(X)), replace=False)]
ell = lambda p: ((p[:, 0] - 525) / 300) ** 2 + ((p[:, 1] - 246) / 40) ** 2   # zonas vistas na máscara do frame 600
GL = {'circle': sub(P[(np.abs(ell(P) - 1) < 0.35) & (P[:, 1] > 205)], 300), 'halfway': sub(P[(P[:, 1] > 285) & (P[:, 1] < 600)], 200),
      'near': sub(P[(P[:, 1] > 600) & (P[:, 0] < 600)], 200)}
NA = len(A)
def unpack(x): return x[:3], x[3], x[4], [x[5 + 4 * i:9 + 4 * i] for i in range(NA + 1)]
def res(x, peso_linhas=5.0):
    C, k1, L, ps = unpack(x); cam = Cam(C, k1); r = []
    for a, uv, p in zip(A, UV, ps):
        q = cam.to_img(p, campo(a, L)); r.append(np.nan_to_num(q - uv, nan=500).ravel())
    p = ps[-1]
    for k, uv in GL.items():
        g = cam.to_pitch(p, uv)
        d = {'circle': np.hypot(g[:, 0], g[:, 1]) - 9.15, 'halfway': g[:, 0], 'near': g[:, 1] + W / 2}[k]
        r.append(peso_linhas * np.clip(d, -20, 20) / np.sqrt(len(uv)) * 10)
    return np.concatenate(r)
anterior = json.load(open(Path(out).parent / 'camara_ancora.json'))
best = None
for cy in (-34, -36, -38, -42):
    for cz in (4, 6, 9):
        for cx in (0, 13):
            C = np.array([cx, cy, cz]); ps = []
            for a in A:
                y, t, r_ = look(C, [45 * (1 if a['baliza'] == 'direita' else -1), 0, 0]); ps.append([y, t, r_, 1300])
            ps.append(anterior['p'])
            x0 = np.r_[C, 0.12, 100.0, np.ravel(ps)]
            lo = np.r_[-40, -60, 2, -0.1, 85, np.tile([-10, -1.5, -1, 300], NA + 1)]
            hi = np.r_[60, -30, 25, 0.5, 115, np.tile([10, 1.5, 1, 5000], NA + 1)]
            try: r = least_squares(res, x0, loss='soft_l1', f_scale=3.0, bounds=(lo, hi), max_nfev=600)
            except Exception: continue
            if best is None or r.cost < best.cost: best = r
C, k1, L, ps = unpack(best.x); cam = Cam(C, k1)
print(f'câmara C={np.round(C, 1)} m, k1={k1:.3f}, comprimento do campo L={L:.1f} m (largura fixa {W})')
for a, uv, p in zip(A, UV, ps):
    e = np.hypot(*(cam.to_img(p, campo(a, L)) - uv).T); print(f"  âncora f{a['frame']} ({a['baliza']}): erro mediano {np.median(e):.1f} px, máximo {e.max():.1f} px")
for k, uv in GL.items():
    g = cam.to_pitch(ps[-1], uv); d = {'circle': np.hypot(g[:, 0], g[:, 1]) - 9.15, 'halfway': g[:, 0], 'near': g[:, 1] + W / 2}[k]
    print(f'  f{F_LINHAS} {k}: erro mediano {np.median(np.abs(d)):.2f} m')
json.dump(dict(nota='Câmara ajustada a pontos marcados à mão nas duas áreas e às linhas do círculo central (fit_camara.py).',
               L=float(L), W=W, C=C.tolist(), k1=float(k1),
               ancoras={str(f): [float(v) for v in p] for f, p in zip([a['frame'] for a in A] + [F_LINHAS], ps)}),
          open(out, 'w'), indent=1, ensure_ascii=False)
print('gravado', out)
