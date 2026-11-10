"""Vídeo com jogadores e bola juntos.
Jogadores: CSV do process.py (amostrado, ex. 5 fps; t em segundos do jogo). Entre amostras, a caixa de cada
trajetória é interpolada linearmente (só se as duas amostras distarem <= 0,5 s), para desenhar a 25 fps.
Bola: CSV do seguir_bola.py (t em segundos do trecho; soma-se --t0 para passar a segundos do jogo).
Uso: python juntar.py <trecho.mp4> <jogadores.csv> <bola.csv> <saida.mp4> --t0 1500"""
import csv, argparse
from collections import defaultdict
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('jogadores'); ap.add_argument('bola'); ap.add_argument('out')
ap.add_argument('--t0', type=float, required=True, help='segundo do jogo onde começa o trecho')
a = ap.parse_args()
COL = {'vermelho': (40, 40, 230), 'rosa': (200, 120, 255), 'arbitro': (0, 230, 255), 'gr': (255, 255, 255)}  # = teams.COL

tracks = defaultdict(list)                                  # id -> [(t, caixa, equipa)]
for r in csv.DictReader(open(a.jogadores)):
    tracks[r['track']].append((float(r['t']), np.array([float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')]), r['equipa']))
for v in tracks.values(): v.sort(key=lambda x: x[0])
ids = {tid: i + 1 for i, tid in enumerate(sorted(tracks, key=lambda k: tracks[k][0][0]))}   # IDs curtos para o vídeo

def jogadores_em(t):
    out = []
    for tid, S in tracks.items():
        if t < S[0][0] - 1e-6 or t > S[-1][0] + 1e-6: continue
        i = np.searchsorted([s[0] for s in S], t)
        if i < len(S) and abs(S[i][0] - t) < 1e-6: out.append((ids[tid], S[i][1], S[i][2])); continue
        (ta, ba, e), (tb, bb, _) = S[i - 1], S[i]
        if tb - ta > 0.5: continue
        out.append((ids[tid], ba + (bb - ba) * (t - ta) / (tb - ta), e))
    return out

bola = {int(r['f']): r for r in csv.DictReader(open(a.bola))}
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'avc1'), fps, (1280, 720))
trail = []; f = 0
while True:
    ok, im = cap.read()
    if not ok: break
    t = a.t0 + f / fps; cnt = defaultdict(int)
    for n, b, e in jogadores_em(t):
        x1, y1, x2, y2 = map(int, b); c = COL.get(e, (200, 200, 200)); cnt[e] += 1
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cv2.putText(im, str(n), (x1, max(45, y1 - 4)), 0, .45, c, 2)
    r = bola.get(f)
    if r and r['x']:
        x, y = int(float(r['x'])), int(float(r['y'])); trail.append((f, x, y))
        cv2.circle(im, (x, y), 11, (0, 255, 255) if r['fonte'] == 'det' else (0, 140, 255), 2)
    trail = [p for p in trail if f - p[0] < 13]
    for p, q in zip(trail, trail[1:]):
        if q[0] - p[0] <= 2: cv2.line(im, p[1:], q[1:], (0, 220, 255), 2)
    mm, ss = divmod(t, 60)
    cv2.rectangle(im, (0, 0), (1280, 30), (25, 25, 25), -1)
    cv2.putText(im, f"{int(mm):02d}:{ss:05.2f}   vermelho {cnt['vermelho']}  rosa {cnt['rosa']}  arbitro {cnt['arbitro']}  GR {cnt['gr']}"
                f"   bola: {'detetada' if r and r['fonte'] == 'det' else 'interpolada' if r and r['x'] else '-'}",
                (10, 21), 0, .6, (255, 255, 255), 1)
    vw.write(im); f += 1
vw.release(); print('vídeo:', a.out, f, 'frames')
