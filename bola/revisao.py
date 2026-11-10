"""Folhas de revisão dos candidatos a bola (união de todas as configurações).
Os candidatos de todos os modelos no mesmo frame a < 8 px uns dos outros juntam-se num só.
Cada candidato aparece como 3 recortes (+ 1 de contexto, 3x mais largo) ampliados do mesmo sítio: frame -4, frame, frame +4 (25 fps, ±0,16 s).
A bola mexe-se ou some entre recortes; marcas do relvado ficam iguais (a câmara mexe pouco em 0,16 s).
Gera <pasta>/candidatos.json e <pasta>/folha_XX.jpg. A revisão é escrita à mão em <pasta>/revisao.json:
  {"bola": [ids dos candidatos que são a bola], "duvida": [ids incertos]}
Uso: python revisao.py <video_trecho> <pasta_comparacao>"""
import sys, json
from pathlib import Path
import cv2, numpy as np

video, d = sys.argv[1], Path(sys.argv[2])
R, Z, DT, POR_FOLHA = 20, 3, 4, 24          # raio do recorte, ampliação, desvio em frames, candidatos por folha
dets = {p.stem[5:]: json.load(open(p))['dets'] for p in sorted(d.glob('dets_*.json'))}
frames = sorted({int(f) for v in dets.values() for f in v})
cands = []                                   # (frame, cx, cy, {config: conf})
for f in frames:
    cs = []
    for cfg, v in dets.items():
        for x in v.get(str(f), []):
            b = x['xyxy']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            m = next((c for c in cs if np.hypot(c[0] - cx, c[1] - cy) < 8), None)
            if m is None: cs.append([cx, cy, {cfg: x['conf']}])
            else: m[2][cfg] = max(m[2].get(cfg, 0), x['conf'])
    cands += [dict(id=len(cands) + i, frame=f, x=round(c[0], 1), y=round(c[1], 1), confs=c[2]) for i, c in enumerate(cs)]
# candidatos só do RT-DETR com conf < 0,15 não se revêem (são ~28 por frame); contam como falsos
cands = [c for c in cands if not (set(c['confs']) == {'rtdetr-l_1280'} and c['confs']['rtdetr-l_1280'] < 0.15)]
for i, c in enumerate(cands): c['id'] = i
json.dump(cands, open(d / 'candidatos.json', 'w'), indent=0)
print(len(cands), 'candidatos em', len(frames), 'frames')

cap = cv2.VideoCapture(video); cache = {}
def frame(f):
    if f not in cache:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); cache[f] = im if ok else np.zeros((720, 1280, 3), np.uint8)
    return cache[f]
def crop(im, x, y):
    x, y = int(round(x)), int(round(y)); p = cv2.copyMakeBorder(im, R, R, R, R, cv2.BORDER_CONSTANT)
    return cv2.resize(p[y:y + 2 * R, x:x + 2 * R], (2 * R * Z, 2 * R * Z), interpolation=cv2.INTER_CUBIC)

for s in range(0, len(cands), POR_FOLHA):
    rows = []
    for c in cands[s:s + POR_FOLHA]:
        tiles = [crop(frame(max(0, c['frame'] + o)), c['x'], c['y']) for o in (-DT, 0, DT)]
        t = tiles[1]; k = R * Z; g = 14                       # cantos a marcar o centro (sem tapar)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cv2.line(t, (k + sx * g, k + sy * g), (k + sx * 2 * g, k + sy * g), (0, 0, 255), 1)
                cv2.line(t, (k + sx * g, k + sy * g), (k + sx * g, k + sy * 2 * g), (0, 0, 255), 1)
        ctx = frame(c['frame']).copy(); cv2.circle(ctx, (int(c['x']), int(c['y'])), 9, (0, 0, 255), 1)
        x, y = int(c['x']), int(c['y']); RC = 3 * R                 # contexto: 3x mais largo, sem ampliação extra
        ctx = cv2.copyMakeBorder(ctx, RC, RC, RC, RC, cv2.BORDER_CONSTANT)[y:y + 2 * RC, x:x + 2 * RC]
        ctx = cv2.resize(ctx, t.shape[1::-1], interpolation=cv2.INTER_AREA)
        sep = np.full((t.shape[0], 3, 3), 255, np.uint8)
        row = np.hstack([tiles[0], sep, t, sep, tiles[2], sep, ctx])
        cv2.putText(row, f"#{c['id']} f{c['frame']}", (4, 16), 0, .5, (255, 255, 255), 2)
        rows.append(row)
    while len(rows) % 2: rows.append(np.zeros_like(rows[0]))
    sheet = np.vstack([np.hstack([rows[i], np.full((rows[i].shape[0], 12, 3), 0, np.uint8), rows[i + 1]])
                       for i in range(0, len(rows), 2)])
    cv2.imwrite(str(d / f'folha_{s // POR_FOLHA:02d}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print('folhas:', (len(cands) + POR_FOLHA - 1) // POR_FOLHA)

# vista geral por frame: frame inteiro com os números dos candidatos (para decidir qual é a bola do jogo)
for f in frames:
    im = frame(f).copy()
    for c in cands:
        if c['frame'] != f: continue
        x, y = int(c['x']), int(c['y']); cv2.circle(im, (x, y), 9, (0, 0, 255), 1)
        cv2.putText(im, str(c['id']), (x + 9, y - 9), 0, .4, (0, 0, 255), 1)
    cv2.putText(im, f'f{f}', (8, 20), 0, .6, (255, 255, 255), 2)
    cv2.imwrite(str(d / f'geral_{f:04d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
