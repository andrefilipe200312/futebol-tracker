"""Calibração do campo (homografia campo->imagem) frame a frame.
1) Frame inicial: homografia a partir de pontos marcados à mão.
2) Cada frame seguinte: movimento da câmara entre frames (SIFT na relva + RANSAC; a câmara roda
   sobre si própria, por isso a relação entre frames é uma homografia), depois
3) refinamento: alinha o modelo do campo de 11 às linhas brancas detetadas (distância truncada,
   nos dois sentidos). A largura e o comprimento do campo são estimados no início.
Uso: python calib.py <pasta_saida_deteção>   (escreve calib.json na mesma pasta)"""
import sys, json
import cv2, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from pitch import grass_mask, white_mask, template

TAU = 12.0          # px: distância máxima contada (robustez a linhas falsas/ocultas)
PPM = 4             # raster do campo (px por metro) para a distância imagem->modelo

def proj(H, P):
    q = np.c_[P, np.ones(len(P))] @ H.T; return q[:, :2] / q[:, 2:3]

def corners(L, W): return np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]], float)

class Model:
    def __init__(s, L, W):
        s.L, s.W = L, W; s.T = template(L, W)
        m = 6; s.off = np.array([L/2 + m, W/2 + m])
        img = np.full((int((W + 2*m) * PPM), int((L + 2*m) * PPM)), 255, np.uint8)
        for p in s.T: cv2.circle(img, tuple(np.round((p + s.off) * PPM).astype(int)), 1, 0, -1)
        s.dt = cv2.distanceTransform(img, cv2.DIST_L2, 3) / PPM   # metros até à linha mais próxima

def residuals(H, M, dti, white_pts, valid, sel=None):
    """a) pontos do modelo projetados -> distância às linhas brancas detetadas (só os que caem em relva visível;
          o conjunto é fixado em `sel` durante a otimização para o vetor ter tamanho constante);
       b) píxeis brancos -> distância à linha do modelo mais próxima (convertida para px pela escala local).
       Interpolação bilinear para a função ser suave."""
    h, w = dti.shape
    q = proj(H, M.T)
    if sel is None:
        ok = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        qi = q.astype(int).clip([0, 0], [w - 1, h - 1]); sel = ok & (valid[qi[:, 1], qi[:, 0]] > 0)
    qs = q[sel]
    ra = np.minimum(map_coordinates(dti, [qs[:, 1], qs[:, 0]], order=1, mode='constant', cval=TAU), TAU)
    pm = proj(np.linalg.inv(H), white_pts)
    ij = (pm + M.off) * PPM
    dm = map_coordinates(M.dt, [ij[:, 1], ij[:, 0]], order=1, mode='constant', cval=99.0)
    e = proj(H, pm + [0.5, 0]) - white_pts; sc = np.hypot(e[:, 0], e[:, 1]) * 2
    rb = np.minimum(dm * sc, TAU)
    return np.r_[ra / np.sqrt(max(1, len(ra))), rb / np.sqrt(max(1, len(rb)))], sel

def frame_data(im, boxes):
    g = grass_mask(im); wm = white_mask(im, g, boxes)
    valid = g.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b); valid[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
    dti = cv2.distanceTransform(255 - wm, cv2.DIST_L2, 3).astype(np.float64)
    ys, xs = np.nonzero(wm); pts = np.c_[xs, ys].astype(float)
    if len(pts) > 4000: pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    return g, wm, valid, dti, pts

def refine(H, M, fd, iters=2):
    """Parametriza H pelos 4 cantos do campo projetados na imagem (bem condicionado)."""
    _, _, valid, dti, pts = fd
    if len(pts) < 50: return H, 99.0, 0
    C = corners(M.L, M.W); c0 = proj(H, C)
    HC = lambda x: cv2.getPerspectiveTransform(np.float32(C), np.float32(x.reshape(4, 2))).astype(float)
    for _ in range(iters):
        sel = residuals(HC(c0), M, dti, pts, valid)[1]
        if sel.sum() < 30: return H, 99.0, int(sel.sum())
        r = least_squares(lambda x: residuals(HC(x), M, dti, pts, valid, sel)[0], c0.ravel(),
                          loss='soft_l1', f_scale=0.05, diff_step=1e-4, max_nfev=150)
        c0 = r.x.reshape(4, 2)
    H = HC(c0)
    res, sel = residuals(H, M, dti, pts, valid); nvis = int(sel.sum())
    # custo médio em px (os dois termos)
    return H, float(np.sqrt((res ** 2).sum() / 2)), nvis

sift = cv2.SIFT_create(2000)
def interframe(a, b, ma, mb):
    """Homografia imagem a -> imagem b, a partir de pontos SIFT na relva (linhas, textura)."""
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), ma)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), mb)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10: return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 12: return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    G, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    return G, int(inl.sum()) if inl is not None else 0

def feat_mask(g, boxes):
    m = cv2.dilate(g, np.ones((9, 9), np.uint8))
    for b in boxes:
        x1, y1, x2, y2 = map(int, b); m[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6] = 0
    return m

if __name__ == '__main__':
    root = Path(sys.argv[1])
    D = [json.loads(l) for l in open(root / 'detections.jsonl')]; D.sort(key=lambda d: d['k'])
    # pontos marcados à mão no primeiro frame do jogo (círculo central e linha do meio), em metros -> píxeis
    init = json.load(open(Path(__file__).parent / 'anchor.json'))
    L, W = init['L'], init['W']
    im0 = cv2.imread(str(root / 'frames' / D[0]['frame'])); boxes0 = [d['xyxy'] for d in D[0]['dets']]
    H0, _ = cv2.findHomography(np.float32(init['pitch']), np.float32(init['image']))
    fd0 = frame_data(im0, boxes0)
    # estima a largura do campo no frame inicial (as duas laterais e o círculo estão visíveis)
    best = None
    for Wc in np.arange(56, 74.1, 2.0):
        M = Model(L, Wc); H, c, n = refine(H0, M, fd0)
        if best is None or c < best[1]: best = (Wc, c, H)
        print(f'W={Wc:.0f} custo {c:.2f}px', flush=True)
    W = best[0]
    for Wc in np.arange(W - 1.5, W + 1.6, 0.5):
        M = Model(L, Wc); H, c, n = refine(best[2], M, fd0)
        if c < best[1]: best = (Wc, c, H)
    W, H = best[0], best[2]; print('largura estimada', W, 'custo', round(best[1], 2), flush=True)
    M = Model(L, W)
    out = [dict(k=D[0]['k'], t=D[0]['t'], H=H.tolist(), cost=best[1], inl=-1, ok=True)]
    prev_im, prev_m = im0, feat_mask(fd0[0], boxes0)
    for d in D[1:]:
        im = cv2.imread(str(root / 'frames' / d['frame'])); boxes = [x['xyxy'] for x in d['dets']]
        fd = frame_data(im, boxes); m = feat_mask(fd[0], boxes)
        G, inl = interframe(prev_im, im, prev_m, m)
        Hp = G @ H if G is not None and inl >= 15 else H
        Hn, c, n = refine(Hp, M, fd)
        ok = c < 4.0
        if ok: H = Hn
        else: H = Hp
        out.append(dict(k=d['k'], t=d['t'], H=H.tolist(), cost=c, inl=inl, ok=bool(ok)))
        print(f"{d['k']} t={d['t']:.1f} inl={inl} custo={c:.2f} vis={n} {'ok' if ok else 'FALHA'}", flush=True)
        prev_im, prev_m = im, m
    json.dump(dict(L=L, W=W, frames=out), open(root / 'calib.json', 'w'))
