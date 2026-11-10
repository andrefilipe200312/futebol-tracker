"""Classificação de cada momento em fases de jogo, do ponto de vista de uma equipa (por omissão o vermelho),
e organização da equipa em cada fase.
Entradas (todas com frames absolutos do jogo): calibração (calibrar_trecho.py), jogadores (process.py), bola (seguir_bola.py).

Posse: equipa com o jogador mais perto da bola a <= POSSE_M. Só muda de dono quando a outra equipa a tem durante
>= TROCA amostras seguidas; sem bola/jogador perto mantém o dono até SEM_DONO_S segundos, depois fica indefinida.
Fases da equipa E (fora das bolas paradas):
  com a bola há < TRANS_S s, ganha ao adversário  -> transicao_ofensiva;   com a bola depois disso -> organizacao_ofensiva
  adversário com a bola há < TRANS_S s, perdida por E -> transicao_defensiva; depois disso -> organizacao_defensiva
Bola parada: bola parada no campo (desloca-se < PARADA_M) durante >= PARADA_S s, seguida de reposição. Tipo pelo sítio
(canto, lançamento, pontapé de baliza, livre). Ofensiva se quem a repõe (jogador mais perto quando a bola volta a
mexer) é da equipa E. O episódio vai do início da paragem até APOS_S s depois da reposição.
Organização por fase: largura, profundidade, compacidade, altura da linha defensiva, distâncias entre linhas,
formação aproximada e mapa de calor, só com momentos com >= MIN_VIS jogadores de campo visíveis da equipa E.
Saída: <saida>/amostras.csv, <saida>/segmentos.csv, <saida>/organizacao.json
Uso: python fases.py <calib.json> <jogadores.csv> <bola.csv> <saida_dir> [--equipa vermelho]"""
import sys, json, csv, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrar_trecho import Cam

ap = argparse.ArgumentParser()
ap.add_argument('calib'); ap.add_argument('jogadores'); ap.add_argument('bola'); ap.add_argument('saida')
ap.add_argument('--equipa', default='vermelho'); ap.add_argument('--fps', type=float, default=25)
ap.add_argument('--posse-m', type=float, default=3.0); ap.add_argument('--troca', type=int, default=2)
ap.add_argument('--sem-dono-s', type=float, default=3.0); ap.add_argument('--trans-s', type=float, default=6.0)
ap.add_argument('--min-posse-s', type=float, default=2.0, help='troca de posse mais curta do que isto = disputa')
ap.add_argument('--parada-m', type=float, default=1.0); ap.add_argument('--parada-s', type=float, default=3.0)
ap.add_argument('--apos-s', type=float, default=8.0); ap.add_argument('--min-vis', type=int, default=7)
a = ap.parse_args()
E = a.equipa; ADV = 'rosa' if E == 'vermelho' else 'vermelho'; EQ = (E, ADV)
out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
K = json.load(open(a.calib)); cam = Cam(K['C'], K['k1']); L, W = K['L'], K['W']; FR = K['frames']
def no_campo(x, y, m=3): return abs(x) <= L / 2 + m and abs(y) <= W / 2 + m

# --- posições no campo (frames absolutos; t do CSV de jogadores em segundos do jogo) ---
J = defaultdict(list)
for r in csv.DictReader(open(a.jogadores)):
    f = int(round(float(r['t']) * a.fps)); f -= f % 10
    if str(f) not in FR: continue
    x, y = cam.to_pitch(np.array(FR[str(f)]['p']), [[float(r['x_pes']), float(r['y_pes'])]])[0]
    if no_campo(x, y): J[f].append((r['equipa'], float(x), float(y)))
B = {}
for r in csv.DictReader(open(a.bola)):
    f = int(r['f'])
    if f % 10 or str(f) not in FR or not r['x']: continue
    x, y = cam.to_pitch(np.array(FR[str(f)]['p']), [[float(r['x']), float(r['y'])]])[0]
    if no_campo(x, y, 2): B[f] = (float(x), float(y))
FS = sorted(int(f) for f in FR); dt = 10 / a.fps

# --- sentido de jogo da equipa E por metade do tempo de análise (o mais recuado perto de que baliza) ---
rec = [min(x for e, x, _ in J[f] if e == E) - min(x for e, x, _ in J[f] if e == ADV) for f in FS
       if sum(e == E for e, _, _ in J[f]) >= a.min_vis and sum(e == ADV for e, _, _ in J[f]) >= a.min_vis]
SENT = 1 if np.median(rec) < 0 else -1             # +1: E defende a esquerda e ataca para a direita
def prof(x): return SENT * x + L / 2                # metros desde a própria linha de golo de E

# --- posse bruta e suavizada ---
bruta = {}
for f in FS:
    if f not in B: bruta[f] = None; continue
    bx, by = B[f]; d = [(np.hypot(x - bx, y - by), e) for e, x, y in J[f] if e in EQ]
    bruta[f] = min(d)[1] if d and min(d)[0] <= a.posse_m else None
dono, cand, ncand, ultimo = None, None, 0, -1e9; posse = {}; ganho = {}
for f in FS:
    r = bruta[f]
    if r is None:
        if dono and (f - ultimo) / a.fps > a.sem_dono_s: dono = None
    elif r == dono: ultimo = f; ncand = 0
    else:
        ncand = ncand + 1 if r == cand else 1; cand = r
        if ncand >= a.troca or dono is None:
            ganho[f] = (dono, r); dono = r; ultimo = f; ncand = 0
    posse[f] = dono
# trocas curtas (< MIN_POSSE_S) entre dois períodos do mesmo dono são disputas: a posse fica com esse dono
def corridas(seq):
    R = []
    for f in FS:
        if R and R[-1][0] == seq[f]: R[-1][2] = f
        else: R.append([seq[f], f, f])
    return R
for _ in range(3):
    R = corridas(posse)
    for i in range(1, len(R) - 1):
        (d0, _, _), (d1, i1, f1), (d2, _, _) = R[i - 1], R[i], R[i + 1]
        if d0 == d2 and d0 is not None and d1 != d0 and (f1 - i1) / a.fps + dt < a.min_posse_s:
            for f in FS:
                if i1 <= f <= f1: posse[f] = d0
ganho = {}; prev = None
for f in FS:
    if posse[f] is not None and posse[f] != prev and (prev is not None or ganho): ganho[f] = (prev, posse[f])
    if posse[f] is not None: prev = posse[f]

# --- bolas paradas: bola parada >= PARADA_S e reposição ---
paradas = []; fb = [f for f in FS if f in B]; i = 0
while i < len(fb):
    j = i
    while j + 1 < len(fb) and fb[j + 1] - fb[j] <= 30 and np.hypot(*(np.array(B[fb[j + 1]]) - B[fb[i]])) < a.parada_m: j += 1
    dur = (fb[j] - fb[i]) / a.fps
    if dur >= a.parada_s and j + 1 < len(fb):
        bx, by = B[fb[i]]; frep = fb[j]
        d = [(np.hypot(x - bx, y - by), e) for e, x, y in J[frep] if e in EQ]
        quem = min(d)[1] if d and min(d)[0] <= 5 else None
        if abs(bx) > L / 2 - 4 and abs(by) > W / 2 - 4: tipo = 'canto'
        elif abs(by) > W / 2 - 2: tipo = 'lançamento'
        elif abs(bx) > L / 2 - 6.5 and abs(by) < 10: tipo = 'pontapé de baliza'
        else: tipo = 'livre'
        if quem: paradas.append(dict(ini=fb[i], rep=frep, fim=frep + int(a.apos_s * a.fps), tipo=tipo, quem=quem, x=bx, y=by))
        i = j + 1
    else: i += 1

# --- fase de cada amostra ---
fase = {}; ultimo_ganho = None
for f in FS:
    if f in ganho: ultimo_ganho = (f, *ganho[f])
    bp = next((p for p in paradas if p['ini'] <= f <= p['fim']), None)
    if bp: fase[f] = ('bola_parada_ofensiva' if bp['quem'] == E else 'bola_parada_defensiva', bp['tipo'])
    elif posse[f] is None: fase[f] = ('indefinida', '')
    else:
        recente = ultimo_ganho and (f - ultimo_ganho[0]) / a.fps < a.trans_s and ultimo_ganho[1] is not None
        if posse[f] == E: fase[f] = ('transicao_ofensiva' if recente and ultimo_ganho[1] == ADV else 'organizacao_ofensiva', '')
        else: fase[f] = ('transicao_defensiva' if recente and ultimo_ganho[1] == E else 'organizacao_defensiva', '')

# --- segmentos (amostras seguidas da mesma fase; buracos de 1 amostra fundidos) ---
MIN_S = {'organizacao_ofensiva': 4, 'organizacao_defensiva': 4, 'transicao_ofensiva': 2, 'transicao_defensiva': 2,
         'bola_parada_ofensiva': 3, 'bola_parada_defensiva': 3}
segs = []
for f in FS:
    ph, tipo = fase[f]
    if segs and segs[-1]['fase'] == ph and f - segs[-1]['fim'] <= 20: segs[-1]['fim'] = f
    else: segs.append(dict(fase=ph, tipo=tipo, ini=f, fim=f))
segs = [s for s in segs if s['fase'] in MIN_S and (s['fim'] - s['ini']) / a.fps >= MIN_S[s['fase']]]

# --- organização de E por fase ---
def kmeans(X, k, it=60, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C, lab
org = {}
for ph in MIN_S:
    fr = [f for f in FS if fase[f][0] == ph and sum(e == E for e, _, _ in J[f]) >= a.min_vis]
    if len(fr) < 5: org[ph] = dict(momentos=len(fr)); continue
    M = defaultdict(list); R = []; todos = []
    for f in fr:
        P = np.array([[x, y] for e, x, y in J[f] if e == E]); dp = np.sort(prof(P[:, 0])); c = P.mean(0)
        M['largura'].append(np.ptp(P[:, 1])); M['profundidade'].append(np.ptp(P[:, 0]))
        M['compacidade'].append(np.mean(np.hypot(*(P - c).T))); M['linha_def'].append(dp[:4].mean())
        M['def_meio'].append(dp[4:-2].mean() - dp[:4].mean() if len(dp) > 6 else np.nan)
        M['meio_ataque'].append(dp[-2:].mean() - dp[4:-2].mean() if len(dp) > 6 else np.nan)
        M['centro'].append(prof(c[0]))
        Q = np.c_[SENT * P[:, 0], SENT * P[:, 1]]; R += (Q - Q.mean(0)).tolist(); todos += Q.tolist()
    R = np.array(R); k = 10 if len(R) >= 40 else min(10, len(R) // 4)
    C, lab = min((kmeans(R, k, seed=s) for s in range(6)), key=lambda c: ((R - c[0][c[1]]) ** 2).sum())
    C = C[np.argsort(C[:, 0])]; g = np.diff(C[:, 0]); o = np.argsort(g)[::-1]
    nc = 3 if len(g) > 2 and g[o[2]] >= 0.7 * g[o[1]] else 2
    esquema = '-'.join(str(len(l)) for l in np.split(np.arange(len(C)), sorted(o[:nc] + 1)))
    bx = np.linspace(-L / 2, L / 2, int(round(L / 5)) + 1); by = np.linspace(-W / 2, W / 2, int(round(W / 5)) + 1)
    T = np.array(todos); Hc = np.histogram2d(T[:, 0], T[:, 1], [bx, by])[0].astype(int)
    org[ph] = dict(momentos=len(fr), segundos=round(len(fr) * dt, 1),
                   **{k2: round(float(np.nanmean(v)), 1) for k2, v in M.items()},
                   esquema=esquema, posicoes=[[round(float(x), 1), round(float(y), 1)] for x, y in C],
                   calor=dict(bins_x=bx.tolist(), bins_y=by.tolist(), n=Hc.tolist()))

with open(out / 'amostras.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['f', 't', 'posse_bruta', 'posse', 'fase', 'tipo', 'bola_x', 'bola_y', f'visiveis_{E}', f'visiveis_{ADV}'])
    for f in FS: w.writerow([f, round(f / a.fps, 1), bruta[f] or '', posse[f] or '', *fase[f], *(np.round(B[f], 1) if f in B else ['', '']),
                             sum(e == E for e, _, _ in J[f]), sum(e == ADV for e, _, _ in J[f])])
with open(out / 'segmentos.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['fase', 'tipo', 'f_ini', 'f_fim', 't_ini', 't_fim', 'duracao_s'])
    for s in segs: w.writerow([s['fase'], s['tipo'], s['ini'], s['fim'], round(s['ini'] / a.fps, 1), round(s['fim'] / a.fps, 1), round((s['fim'] - s['ini']) / a.fps + dt, 1)])
json.dump(dict(equipa=E, adversario=ADV, sentido='ataca para a direita' if SENT > 0 else 'ataca para a esquerda',
               parametros={k: v for k, v in vars(a).items() if k not in ('calib', 'jogadores', 'bola', 'saida')},
               bolas_paradas=paradas, organizacao=org, campo=dict(L=L, W=W)), open(out / 'organizacao.json', 'w'), ensure_ascii=False)
c = Counter(fase[f][0] for f in FS)
print(f'{E} {"ataca para a direita" if SENT > 0 else "ataca para a esquerda"}; amostras por fase:', dict(c))
print('segmentos por fase:', dict(Counter(s['fase'] for s in segs)), '| bolas paradas:', Counter((p['tipo'], p['quem']) for p in paradas))
