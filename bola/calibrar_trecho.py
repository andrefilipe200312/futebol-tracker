"""Calibração imagem <-> campo (metros) nos frames analisados de um trecho, com o modelo físico de câmara
(../camera.py): posição fixa da câmara + distorção da lente + largura do campo, ajustadas uma vez num frame-âncora
(fit_ancora); em cada frame só mudam rumo, inclinação, rotação e focal (a câmara roda e faz zoom).
Por frame: previsão a partir do frame anterior + movimento da câmara entre frames (G, guardado por
detetar_bola.py), depois afinação encostando as linhas do modelo do campo às linhas brancas da imagem.
Campo: metros, origem no centro, x ao comprimento (para a direita da imagem na âncora), y para a lateral do fundo.
Saída: <saida>.json {L, W, k1, C, frames: {f: {p: [rumo, incl, rot, focal], custo, n, ok}}}.
Uso: python calibrar_trecho.py <trecho.mp4> <dets.jsonl> <camara.json (fit_camara.py)> <saida.json>"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import camera
from camera import homography
from pitch import grass_mask, white_mask, template

TAU = 12.0
class Cam:
    """Projeção campo (m) <-> imagem (px), com a posição C, a lente k1 e a largura W fixas."""
    def __init__(s, C, k1): s.C = np.asarray(C, float); s.k1 = k1
    def und(s, uv):
        uv = np.asarray(uv, float); x = (uv[:, 0] - camera.PP[0]) / camera.FD; y = (uv[:, 1] - camera.PP[1]) / camera.FD
        r = 1 + s.k1 * (x * x + y * y); return np.c_[x * r * camera.FD + camera.PP[0], y * r * camera.FD + camera.PP[1]]
    def dist(s, uv):
        uv = np.asarray(uv, float); q = uv.copy()
        for _ in range(12):
            x = (q[:, 0] - camera.PP[0]) / camera.FD; y = (q[:, 1] - camera.PP[1]) / camera.FD; r = 1 + s.k1 * (x * x + y * y)
            q = np.c_[(uv[:, 0] - camera.PP[0]) / r + camera.PP[0], (uv[:, 1] - camera.PP[1]) / r + camera.PP[1]]
        return q
    def H(s, p): return homography(np.r_[s.C, p])
    def to_img(s, p, P):
        q = np.c_[P, np.ones(len(P))] @ s.H(p).T; front = q[:, 2] > 1e-6
        out = np.full((len(P), 2), np.nan); out[front] = s.dist(q[front, :2] / q[front, 2:3]); return out
    def to_pitch(s, p, uv):
        q = np.c_[s.und(uv), np.ones(len(uv))] @ np.linalg.inv(s.H(p)).T; return q[:, :2] / q[:, 2:3]

def ajusta(cam, p0, T, dti, valid):
    """Afina [rumo, incl, rot, focal]: pontos das linhas do modelo -> distância às linhas brancas (truncada)."""
    h, w = dti.shape
    def sel(p):
        q = cam.to_img(p, T); ok = np.isfinite(q[:, 0]) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        qi = np.nan_to_num(q).astype(int).clip([0, 0], [w - 1, h - 1]); return ok & (valid[qi[:, 1], qi[:, 0]] > 0)
    p = np.asarray(p0, float)
    for _ in range(2):
        S = sel(p)
        if S.sum() < 40: return p, 99.0, int(S.sum())
        TS = T[S]
        def res(x):
            q = np.nan_to_num(cam.to_img(x, TS), nan=-1e4)
            return np.minimum(map_coordinates(dti, [q[:, 1], q[:, 0]], order=1, mode='constant', cval=TAU), TAU) / np.sqrt(len(TS))
        p = least_squares(res, p, loss='soft_l1', f_scale=0.05, x_scale=[0.01, 0.01, 0.01, 20.0], max_nfev=100).x
    S = sel(p); q = cam.to_img(p, T[S])
    d = np.minimum(map_coordinates(dti, [q[:, 1], q[:, 0]], order=1, mode='constant', cval=TAU), TAU)
    return p, float(np.median(d)), int(S.sum())

def dados(im, boxes):
    g = grass_mask(im); wm = white_mask(im, g, boxes); valid = g.copy()
    for x1, y1, x2, y2 in boxes: valid[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = 0
    return cv2.distanceTransform(255 - wm, cv2.DIST_L2, 3).astype(np.float64), valid

def prever(cam, p, G):
    """Parâmetros no frame seguinte a partir do movimento da câmara G (imagem anterior -> atual)."""
    T = np.array([[x, y] for x in np.linspace(-45, 45, 10) for y in np.linspace(-30, 30, 7)])
    q = cam.to_img(p, T); ok = np.isfinite(q[:, 0]) & (q[:, 0] > -200) & (q[:, 0] < 1480) & (q[:, 1] > 100) & (q[:, 1] < 820)
    if ok.sum() < 6: return p
    alvo = cv2.perspectiveTransform(q[ok].reshape(-1, 1, 2).astype(np.float32), G).reshape(-1, 2)
    return least_squares(lambda x: (cam.to_img(x, T[ok]) - alvo).ravel(), p, x_scale=[0.01, 0.01, 0.01, 20.0]).x

if __name__ == '__main__':
    video, dets, cam_json, out = sys.argv[1:5]
    K = json.load(open(cam_json)); L, W = K['L'], K['W']; cam = Cam(K['C'], K['k1'])
    D = {d['f']: d for d in (json.loads(l) for l in open(dets) if l.strip())}; FS = sorted(D)
    ANC = {int(f): np.array(p) for f, p in K['ancoras'].items()}; AF = sorted(ANC)
    T = template(L, W, step=0.5)
    cap = cv2.VideoCapture(video)
    def ler(f): cap.set(cv2.CAP_PROP_POS_FRAMES, f); return cap.read()[1]
    cand = {f: [] for f in FS}                                   # f -> [(ok, custo, distância à âncora, p)]
    for fa in AF:
        cand[fa].append((True, 0.0, 0, ANC[fa]))
        i0 = FS.index(fa); prox = [g for g in AF if g > fa]; ant = [g for g in AF if g < fa]
        fim = FS.index(prox[0]) if prox else len(FS) - 1; ini = FS.index(ant[-1]) if ant else 0
        for ordem, sentido in ((FS[i0 + 1:fim + 1], 1), (FS[ini:i0][::-1], -1)):   # até à âncora seguinte/anterior
            p = ANC[fa].copy(); prevf = fa
            for f in ordem:
                G = None                                          # G guardado é (anterior -> atual) no tempo
                if sentido > 0 and D[f]['G']: G = np.array(D[f]['G']).reshape(3, 3)
                if sentido < 0 and D[prevf]['G']: G = np.linalg.inv(np.array(D[prevf]['G']).reshape(3, 3))
                pp = prever(cam, p, G) if G is not None else p
                dti, valid = dados(ler(f), D[f]['persons'])
                pn, c, n = ajusta(cam, pp, T, dti, valid)
                ok = c < 3.0 and n >= 60; p = pn if ok else pp
                cand[f].append((bool(ok), round(c, 2), abs(f - fa), p)); prevf = f
        print(f'âncora f{fa}: propagada', flush=True)
    res = {}
    for f in FS:                                                  # melhor: afinado e com menor custo; senão, o da âncora mais próxima
        C = sorted(cand[f], key=lambda x: (not x[0], x[1] if x[0] else x[2]))[0]
        res[f] = dict(p=[float(v) for v in C[3]], custo=C[1], ok=C[0], dist_ancora=C[2])
    nok = sum(r['ok'] for r in res.values()); print(f'{nok}/{len(FS)} frames afinados pelas linhas; restantes previstos pelo movimento da câmara')
    json.dump(dict(L=L, W=W, k1=cam.k1, C=cam.C.tolist(), frames={str(f): res[f] for f in FS}), open(out, 'w'))
    print('FEITO', out, flush=True)
