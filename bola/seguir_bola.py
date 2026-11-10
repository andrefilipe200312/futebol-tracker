"""Passo 2 do tracking da bola: a partir das deteções guardadas (detetar_bola.py). Rápido: dá para afinar.
1. Referencial fixo: acumula o movimento da câmara (homografias entre frames) e passa cada candidato para as
   coordenadas do 1.º frame. Aí as marcas do relvado ficam paradas e a bola mexe-se.
2. Liga candidatos em trajetórias (previsão de velocidade constante, associação gulosa por distância).
3. Descarta trajetórias paradas (marcas, pontos fixos) e curtas; cada frame fica com a trajetória de maior
   pontuação (soma das confianças, com bónus para movimento). Buracos curtos dentro da trajetória são interpolados.
Saída: <saida>.csv (f, t, x, y, fonte=det|interp, track, conf) e, com --video, <saida>.mp4.
Antes de tudo, cada deteção passa um filtro de tamanho e forma relativo à altura dos jogadores no mesmo y.
Uso: python seguir_bola.py <dets.jsonl> <saida> [--video <trecho.mp4>] [--verdade verdade.json] [--candidatos]"""
import json, csv, argparse
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dets'); ap.add_argument('out')
ap.add_argument('--video'); ap.add_argument('--verdade')
ap.add_argument('--conf-novo', type=float, default=0.05, help='conf mínima para começar uma trajetória')
ap.add_argument('--max-lado', type=float, default=30, help='caixas maiores (px) não são bola')
ap.add_argument('--gate', type=float, default=30, help='distância máxima (px) a 1 frame; +10 px por frame de buraco')
ap.add_argument('--max-buraco', type=int, default=12, help='frames sem deteção até a trajetória acabar')
ap.add_argument('--min-hits', type=int, default=4)
ap.add_argument('--parado', type=float, default=3.0, help='dispersão (px) abaixo da qual uma trajetória longa é fixa')
ap.add_argument('--tam', default='0.12,0.26', help='lado maior da caixa / altura esperada de um jogador no mesmo y (min,max)')
ap.add_argument('--min-lado', type=float, default=0.11, help='lado menor / altura esperada de um jogador (mínimo)')
ap.add_argument('--forma', type=float, default=1.6, help='razão máxima largura/altura (ou altura/largura)')
ap.add_argument('--sem-filtro-tamanho', action='store_true')
ap.add_argument('--candidatos', action='store_true', help='desenha no vídeo os candidatos rejeitados')
a = ap.parse_args()
TMIN, TMAX = map(float, a.tam.split(','))

def altura_jogador(persons):
    """Altura esperada de um jogador em função do y dos pés (reta ajustada às pessoas do frame), ou None."""
    P = np.array([[b[3], b[3] - b[1]] for b in persons if b[3] - b[1] > 15])
    if len(P) < 4: return None
    k, c = np.polyfit(P[:, 0], P[:, 1], 1)
    return lambda y: max(10.0, k * y + c)

def tamanho_ok(b, alt):
    """Filtro de tamanho e forma: uma bola tem ~22 cm, ~1/8 da altura de um jogador ao mesmo y, e é redonda
    (marcas do relvado e linhas são achatadas). Limites medidos nas deteções revistas à mão (33/35 bolas, 10/65 falsos)."""
    w, h = b[2] - b[0], b[3] - b[1]
    if max(w, h) / max(1, min(w, h)) > a.forma: return False
    if alt is None: return True
    e = alt(b[3])
    return TMIN <= max(w, h) / e <= TMAX and min(w, h) / e >= a.min_lado

L = [json.loads(l) for l in open(a.dets) if l.strip()]
FS = [d['f'] for d in L]                                # frames analisados
PASSO = int(np.median(np.diff(FS))) if len(FS) > 1 else 1
FPS = next(d['f'] / d['t'] for d in L if d['t'] > 0)
F = list(range(FS[0], FS[-1] + 1)); T = {f: round(f / FPS, 3) for f in F}   # saída em todos os frames
# parâmetros em frames que dependem do espaçamento entre frames analisados
MAX_BURACO = max(a.max_buraco, 2 * PASSO); MIN_HITS = a.min_hits if PASSO <= 2 else 3; N_FIXA = 8 if PASSO <= 2 else 5
# homografia acumulada: frame -> referencial (1.º frame). Se o movimento falhar, assume câmara parada.
H = np.eye(3); Href = {}
for i, d in enumerate(L):
    if i and d['G']: H = H @ np.linalg.inv(np.array(d['G']).reshape(3, 3))
    Href[d['f']] = H.copy()
for fa, fb in zip(FS, FS[1:]):                          # frames não analisados: homografia interpolada
    for f in range(fa + 1, fb): Href[f] = Href[fa] + (Href[fb] - Href[fa]) * (f - fa) / (fb - fa)
def to_ref(f, x, y): return cv2.perspectiveTransform(np.float32([[[x, y]]]), Href[f])[0, 0]
def from_ref(f, p): return cv2.perspectiveTransform(np.float32([[p]]), np.linalg.inv(Href[f]))[0, 0]

# 2. trajetórias
tracks = []; alive = []; alt = None; n_tam = 0
for d in L:
    f = d['f']
    cands = []
    alt = altura_jogador(d['persons']) or alt          # sem jogadores suficientes: usa a reta do frame anterior
    for x1, y1, x2, y2, c in d['balls']:
        if max(x2 - x1, y2 - y1) > a.max_lado: continue
        if not a.sem_filtro_tamanho and not tamanho_ok((x1, y1, x2, y2), alt): n_tam += 1; continue
        cands.append((to_ref(f, (x1 + x2) / 2, (y1 + y2) / 2), c, ((x1 + x2) / 2, (y1 + y2) / 2)))
    pairs = []
    for i, t in enumerate(alive):
        gap = f - t['pts'][-1][0]; pred = t['pts'][-1][1] + t['v'] * gap
        for j, (p, c, _) in enumerate(cands):
            dist = np.linalg.norm(p - pred)
            if dist <= a.gate + 10 * (gap - 1): pairs.append((dist, i, j))
    ti, cj = set(), set()
    for dist, i, j in sorted(pairs):
        if i in ti or j in cj: continue
        t = alive[i]; p, c, img = cands[j]; gap = f - t['pts'][-1][0]
        v = (p - t['pts'][-1][1]) / gap
        t['v'] = v if len(t['pts']) == 1 else 0.5 * t['v'] + 0.5 * v
        t['pts'].append((f, p, c, img)); ti.add(i); cj.add(j)
    for j, (p, c, img) in enumerate(cands):
        if j not in cj and c >= a.conf_novo:
            t = dict(id=len(tracks), pts=[(f, p, c, img)], v=np.zeros(2)); tracks.append(t); alive.append(t)
    alive = [t for t in alive if f - t['pts'][-1][0] <= MAX_BURACO]

# 3. pontuação e escolha
for t in tracks:
    P = np.array([p for _, p, _, _ in t['pts']]); n = len(P)
    t['spread'] = float(np.max(np.linalg.norm(P - np.median(P, 0), axis=1))) if n else 0
    t['caminho'] = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if n > 1 else 0
    t['fixa'] = n >= N_FIXA and t['spread'] < a.parado
# trajetórias curtas no mesmo sítio do campo que uma fixa (±10 s, por causa do desvio acumulado) também são fixas
FX = [(np.mean([f for f, _, _, _ in t['pts']]), np.median([p for _, p, _, _ in t['pts']], 0)) for t in tracks if t['fixa']]
for t in tracks:
    if t['fixa'] or t['spread'] > a.parado: continue
    fm = np.mean([f for f, _, _, _ in t['pts']]); pm = np.median([p for _, p, _, _ in t['pts']], 0)
    t['fixa'] = any(abs(fm - g) <= 250 and np.linalg.norm(pm - q) <= 5 for g, q in FX)
for t in tracks:
    n = len(t['pts'])
    mexe = min(1.0, t['caminho'] / 40)                 # bónus de movimento (até 40 px percorridos)
    t['soma'] = sum(c for _, _, c, _ in t['pts'])
    t['score'] = 0 if (t['fixa'] or n < MIN_HITS) else t['soma'] * (0.3 + 0.7 * mexe)
JAN = max(12, 2 * PASSO)
best = {}                                               # frame -> (score, track, x, y, fonte, conf)
for t in tracks:
    if t['score'] <= 0: continue
    pts = t['pts']
    for (f0, p0, c0, img0), (f1, p1, c1, img1) in zip(pts, pts[1:] + [pts[-1]]):
        for f in range(f0, f1 if f1 > f0 else f0 + 1):
            if f == f0: x, y = img0; fonte, c = 'det', c0
            else: x, y = from_ref(f, p0 + (p1 - p0) * (f - f0) / (f1 - f0)); fonte, c = 'interp', 0
            # pontuação local: confianças desta trajetória a ±JAN frames (uma trajetória longa noutro sítio não ganha
            # a uma deteção forte e recente)
            s = sum(cc for g, _, cc, _ in pts if abs(g - f) <= JAN) * t['score'] / max(1e-9, t['soma'])
            if f not in best or s > best[f][0]: best[f] = (s, t['id'], float(x), float(y), fonte, c)

with open(a.out + '.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['f', 't', 'x', 'y', 'fonte', 'track', 'conf'])
    for f in F:
        if f in best: s, tid, x, y, fonte, c = best[f]; w.writerow([f, T[f], round(x, 1), round(y, 1), fonte, tid, c])
        else: w.writerow([f, T[f], '', '', 'nada', '', ''])
print(f'1 frame analisado em cada {PASSO}' if PASSO > 1 else 'todos os frames analisados')
nf = len(F); nd = sum(1 for v in best.values() if v[4] == 'det'); ni = len(best) - nd
print(f'{nf} frames: bola em {len(best)} ({100*len(best)/nf:.0f}%; {nd} deteção, {ni} interpolada); '
      f'{n_tam} deteções rejeitadas pelo tamanho/forma; {len(tracks)} trajetórias, {sum(t["fixa"] for t in tracks)} fixas descartadas, {sum(t["score"] > 0 for t in tracks)} usadas')

# avaliação contra a revisão manual (frames revistos: {frame: [x, y]} ou {frame: null} se a bola não se vê)
if a.verdade:
    V = {f: g for f, g in json.load(open(a.verdade)).items() if int(f) in T}; ok = fp = fn = tn = 0
    for f, g in V.items():
        f = int(f); o = best.get(f)
        if g is None: tn += o is None; fp += o is not None
        elif o is None: fn += 1
        elif np.hypot(o[2] - g[0], o[3] - g[1]) <= 12: ok += 1
        else: fp += 1; fn += 1
    nv = sum(g is not None for g in V.values())
    print(f'verdade: {len(V)} frames, bola visível em {nv}: certo {ok}/{nv} ({100*ok/max(1,nv):.0f}%), '
          f'falsos {fp}, sem bola corretamente {tn}/{len(V)-nv}')

if a.video:
    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, F[0])
    vw = cv2.VideoWriter(a.out + '.mp4', cv2.VideoWriter_fourcc(*'avc1'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))
    D = {d['f']: d for d in L}; fixas = {(f, round(img[0]), round(img[1])) for t in tracks if t['fixa'] for f, _, _, img in t['pts']}
    trail = []
    for f in F:
        ok, im = cap.read()
        if not ok: break
        for x1, y1, x2, y2, c in (D[f]['balls'] if a.candidatos and f in D else []):   # candidatos: cinzento; fixos: X
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if (f, round((x1 + x2) / 2), round((y1 + y2) / 2)) in fixas:
                cv2.drawMarker(im, (cx, cy), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 10, 1)
            else: cv2.circle(im, (cx, cy), 7, (170, 170, 170), 1)
        if f in best:
            _, tid, x, y, fonte, c = best[f]; trail.append((f, to_ref(f, x, y)))
            cor = (0, 255, 255) if fonte == 'det' else (0, 140, 255)
            cv2.circle(im, (int(x), int(y)), 11, cor, 2)
        trail = [(g, p) for g, p in trail if f - g < 25]
        pts = [from_ref(f, p).astype(int) for _, p in trail]
        for p, q in zip(pts, pts[1:]): cv2.line(im, tuple(map(int, p)), tuple(map(int, q)), (0, 220, 255), 1)
        cv2.rectangle(im, (0, 0), (1280, 30), (25, 25, 25), -1)
        txt = f"t={T[f]:.2f}s  bola: " + (f"{best[f][4]} (traj. {best[f][1]})" if f in best else '-') + \
              "   amarelo=detetada  laranja=interpolada" + ("  cinzento=candidatos  X=marcas fixas" if a.candidatos else "")
        cv2.putText(im, txt, (10, 21), 0, .55, (255, 255, 255), 1)
        vw.write(im)
    vw.release(); print('vídeo:', a.out + '.mp4')
