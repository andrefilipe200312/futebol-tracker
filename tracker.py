"""Seguimento entre frames (só pessoas em campo: vermelho, rosa, árbitro, GR).
- Movimento da câmara entre frames: fluxo ótico (Lucas-Kanade) em pontos fora das pessoas + homografia RANSAC.
  A câmara só roda/faz zoom, por isso a relação entre frames é uma homografia.
- Previsão: caixa transportada pela homografia da câmara + velocidade própria.
- Associação: algoritmo húngaro com custo de distância (normalizada pela altura) + IoU + tamanho.
  Proibido juntar classes diferentes: vermelho, rosa e preto (árbitro/GR, que vestem igual), assim que a
  trajetória tem classe estável (evita o ID saltar de um jogador para o árbitro num cruzamento).
- Rótulo final de cada trajetória por maioria; GR se foi GR em >= GR_SHARE dos frames.
Pensado para 5 fps ou mais (a 1 fps os jogadores mexem-se demasiado entre frames)."""
import cv2, numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

TRACKED = {'vermelho', 'rosa', 'arbitro', 'gr'}
TEAMS = {'vermelho', 'rosa'}
def cls(l):
    """Classe visual: árbitro e GR vestem ambos de preto."""
    return 'preto' if l in ('arbitro', 'gr') else l
GR_SHARE = 0.3

def camera_motion(prev_gray, gray, prev_boxes):
    """Homografia prev -> atual. None se falhar."""
    if prev_gray is None: return None
    m = np.full(prev_gray.shape, 255, np.uint8); m[:int(m.shape[0] * .04)] = 0
    for x1, y1, x2, y2 in prev_boxes: m[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = 0
    p = cv2.goodFeaturesToTrack(prev_gray, 800, 0.01, 12, mask=m)
    if p is None or len(p) < 30: return None
    q, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p, None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 30: return None
    G, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 2.0)
    if G is None or inl.sum() < 25: return None
    return G

def warp_box(b, G):
    if G is None: return np.asarray(b, float)
    c = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), G).reshape(-1, 2)
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])

def iou(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])); it = wh.prod()
    return it / max(1e-6, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - it)

class Track:
    def __init__(s, tid, box, lab, k):
        s.id = tid; s.box = np.asarray(box, float); s.v = np.zeros(2); s.miss = 0; s.labels = Counter([lab]); s.frames = [k]
    def team(s):
        """Classe estável (vermelho/rosa/preto): >= 2 votos e >= 70 % dos frames. None se ainda incerta."""
        c = {}
        for k, v in s.labels.items():
            if k in TRACKED: c[cls(k)] = c.get(cls(k), 0) + v
        if not c: return None
        k, v = max(c.items(), key=lambda x: x[1])
        return k if v >= 2 and v >= 0.7 * sum(c.values()) else None

class Tracker:
    def __init__(s, fps, max_miss_s=1.5, min_frames=3):
        s.fps = fps; s.tracks = []; s.dead = []; s.next = 1
        s.max_miss = int(round(max_miss_s * fps)); s.min_frames = min_frames; s.gain = 0.5
    def step(s, k, dets, G):
        """dets: lista de (xyxy, rótulo do frame). Devolve lista de (track_id, xyxy, rótulo do frame) para as associadas."""
        dets = [(np.asarray(b, float), l) for b, l in dets if l in TRACKED]
        for t in s.tracks:
            t.box = warp_box(t.box, G)
            if G is not None: t.v = (G[:2, :2] @ t.v) if abs(np.linalg.det(G[:2, :2]) - 1) < .5 else t.v
            t.box = t.box + np.tile(t.v, 2); t.miss += 1
        out = []; used = set()
        if s.tracks and dets:
            Cm = np.full((len(s.tracks), len(dets)), 1e6)
            for i, t in enumerate(s.tracks):
                h = t.box[3] - t.box[1]; tt = t.team()
                for j, (b, l) in enumerate(dets):
                    if tt and cls(l) != tt: continue
                    hh = b[3] - b[1]
                    d = np.hypot(*((t.box[:2] + t.box[2:]) / 2 - (b[:2] + b[2:]) / 2)) / max(15, (h + hh) / 2)
                    gate = 1.2 + 0.4 * min(t.miss - 1, 3)
                    if d > gate or min(h, hh) / max(h, hh) < 0.5: continue
                    Cm[i, j] = 0.5 * (1 - iou(t.box, b)) + 0.5 * d + 0.3 * abs(np.log(hh / h))
            for i, j in zip(*linear_sum_assignment(Cm)):
                if Cm[i, j] >= 1e5: continue
                t = s.tracks[i]; b, l = dets[j]
                c_new = (b[:2] + b[2:]) / 2; c_old = (t.box[:2] + t.box[2:]) / 2
                t.v = (1 - s.gain) * t.v + s.gain * (c_new - c_old + t.v) / max(1, t.miss)
                t.box = b; t.miss = 0; t.labels[l] += 1; t.frames.append(k); used.add(j)
                out.append((t.id, b, l))
        for j, (b, l) in enumerate(dets):
            if j in used: continue
            t = Track(s.next, b, l, k); s.next += 1; s.tracks.append(t); out.append((t.id, b, l))
        keep = []
        for t in s.tracks: (keep if t.miss <= s.max_miss else s.dead).append(t)
        s.tracks = keep
        return out
    def final_labels(s):
        """Rótulo de cada trajetória para todo o trecho. Trajetórias com < min_frames frames -> None (ruído)."""
        res = {}
        for t in s.tracks + s.dead:
            n = sum(t.labels.values())
            if len(t.frames) < s.min_frames: res[t.id] = None; continue
            if t.labels['gr'] >= GR_SHARE * n: res[t.id] = 'gr'; continue
            res[t.id] = max(t.labels.items(), key=lambda x: x[1])[0]
        return res
