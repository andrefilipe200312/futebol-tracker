"""Análise tática de um trecho: jogadores (process.py) e bola (seguir_bola.py) passados para metros com a
calibração (calibrar_trecho.py). Só conta jogadores visíveis: a câmara não mostra o campo inteiro, por isso as
métricas de forma usam só frames com >= MIN_VIS jogadores de campo dessa equipa.
Métricas: 1) posições médias e formação aproximada; 2) mapas de calor (equipas e bola); 3) largura, profundidade,
compacidade e área ao longo do tempo, com e sem bola; 4) altura da linha defensiva e distância entre linhas;
5) território (terço onde está a bola) e posse aproximada (equipa com o jogador mais perto da bola, a <= 3 m).
Saída: <saida>.json com tudo (para a página de resultados).
Uso: python analise_tatica.py <calib.json> <jogadores.csv> <bola.csv> <saida.json> --t0 1500"""
import sys, json, csv, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrar_trecho import Cam

ap = argparse.ArgumentParser()
ap.add_argument('calib'); ap.add_argument('jogadores'); ap.add_argument('bola'); ap.add_argument('out')
ap.add_argument('--t0', type=float, required=True); ap.add_argument('--fps', type=float, default=25)
ap.add_argument('--min-vis', type=int, default=8); ap.add_argument('--posse-m', type=float, default=3.0)
a = ap.parse_args()
K = json.load(open(a.calib)); cam = Cam(K['C'], K['k1']); L, W = K['L'], K['W']; FR = K['frames']
EQ = ('vermelho', 'rosa')
def no_campo(x, y): return abs(x) <= L / 2 + 3 and abs(y) <= W / 2 + 3

# posições no campo por frame analisado
J = defaultdict(list)                                        # f -> [(equipa, x, y, track)]
for r in csv.DictReader(open(a.jogadores)):
    f = int(round((float(r['t']) - a.t0) * a.fps)); f -= f % 10
    if str(f) not in FR: continue
    x, y = cam.to_pitch(np.array(FR[str(f)]['p']), [[float(r['x_pes']), float(r['y_pes'])]])[0]
    if no_campo(x, y): J[f].append((r['equipa'], float(x), float(y), r['track']))
B = {}                                                       # f -> (x, y, fonte)
for r in csv.DictReader(open(a.bola)):
    f = int(r['f'])
    if str(f) not in FR or not r['x']: continue
    x, y = cam.to_pitch(np.array(FR[str(f)]['p']), [[float(r['x']), float(r['y'])]])[0]
    if no_campo(x, y): B[f] = (float(x), float(y), r['fonte'])
FS = sorted(J)

# sentido de jogo: a equipa cujo jogador mais recuado está mais vezes perto da baliza da esquerda defende a esquerda
mais_rec = {e: [min(q[1] for q in J[f] if q[0] == e) for f in FS if sum(q[0] == e for q in J[f]) >= a.min_vis] for e in EQ}
defende_esq = min(EQ, key=lambda e: np.median(mais_rec[e]))
SENT = {e: (1 if e == defende_esq else -1) for e in EQ}       # +1: ataca para a direita (x crescente)
def prof(e, x): return SENT[e] * x + L / 2                   # metros desde a própria linha de golo

# posse aproximada
posse = {}
for f in FS:
    if f not in B: continue
    bx, by, _ = B[f]; d = [(np.hypot(x - bx, y - by), e) for e, x, y, _ in J[f] if e in EQ]
    if d:
        dm, e = min(d); posse[f] = e if dm <= a.posse_m else None

# séries por frame e por equipa
serie = {e: [] for e in EQ}
for f in FS:
    for e in EQ:
        P = np.array([[x, y] for q, x, y, _ in J[f] if q == e])
        if len(P) < a.min_vis: continue
        dpt = np.sort(prof(e, P[:, 0])); c = P.mean(0)
        defesa = dpt[:4].mean(); ataque = dpt[-2:].mean(); meio = dpt[4:-2].mean() if len(dpt) > 6 else np.nan
        serie[e].append(dict(t=round(a.t0 + f / a.fps, 1), f=f, n=len(P), largura=round(float(np.ptp(P[:, 1])), 1),
            profundidade=round(float(np.ptp(P[:, 0])), 1), compacidade=round(float(np.mean(np.hypot(*(P - c).T))), 1),
            area=round(float(ConvexHull(P).volume), 0), linha_def=round(float(defesa), 1),
            def_meio=round(float(meio - defesa), 1), meio_ataque=round(float(ataque - meio), 1),
            centro_x=round(float(c[0]), 1), posse=('com' if posse.get(f) == e else 'sem' if posse.get(f) else None)))

def resumo(S, chave, filtro=None):
    v = [s[chave] for s in S if (filtro is None or s['posse'] == filtro) and s[chave] == s[chave]]
    return dict(media=round(float(np.mean(v)), 1), n=len(v)) if v else None
resumos = {e: {k: dict(todos=resumo(serie[e], k), com=resumo(serie[e], k, 'com'), sem=resumo(serie[e], k, 'sem'))
               for k in ('largura', 'profundidade', 'compacidade', 'area', 'linha_def', 'def_meio', 'meio_ataque')} for e in EQ}

# formação aproximada: posições relativas ao centro da equipa (no sentido de ataque), 10 grupos (k-médias)
def kmeans(X, k, it=60, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C, lab
formacao = {}
for e in EQ:
    R = []
    for f in FS:
        P = np.array([[x, y] for q, x, y, _ in J[f] if q == e])
        if len(P) < a.min_vis: continue
        Q = np.c_[SENT[e] * P[:, 0], SENT[e] * P[:, 1]]; R += (Q - Q.mean(0)).tolist()
    R = np.array(R); C, lab = min((kmeans(R, 10, seed=s) for s in range(8)), key=lambda c: ((R - c[0][c[1]]) ** 2).sum())
    # linhas: corta nos 2 maiores intervalos de profundidade (3 linhas); 4 linhas se o 3.º intervalo for >= 70 % do 2.º
    C = C[np.argsort(C[:, 0])]; gaps = np.diff(C[:, 0]); ordem = np.argsort(gaps)[::-1]
    ncortes = 3 if gaps[ordem[2]] >= 0.7 * gaps[ordem[1]] else 2
    cortes = sorted(ordem[:ncortes] + 1); linhas = np.split(np.arange(10), cortes)
    formacao[e] = dict(posicoes=[[round(float(x), 1), round(float(y), 1)] for x, y in C],
                       esquema='-'.join(str(len(l)) for l in linhas), intervalos_m=[round(float(g), 1) for g in gaps])

# mapas de calor (bins de 5 m) e território
bx = np.linspace(-L / 2, L / 2, int(round(L / 5)) + 1); by = np.linspace(-W / 2, W / 2, int(round(W / 5)) + 1)
calor = {e: np.histogram2d([x for f in FS for q, x, y, _ in J[f] if q == e], [y for f in FS for q, x, y, _ in J[f] if q == e],
                           [bx, by])[0].astype(int).tolist() for e in EQ}
calor['bola'] = np.histogram2d([v[0] for v in B.values()], [v[1] for v in B.values()], [bx, by])[0].astype(int).tolist()
terc = np.digitize([v[0] for v in B.values()], [-L / 6, L / 6])
territorio = {nome: round(100 * float(np.mean(terc == i)), 1) for i, nome in enumerate(('terco_esq', 'terco_meio', 'terco_dir'))}
pv = [v for v in posse.values()]
posse_pct = {e: round(100 * pv.count(e) / max(1, len(pv)), 1) for e in EQ}; posse_pct['disputada'] = round(100 * pv.count(None) / max(1, len(pv)), 1)

json.dump(dict(trecho=dict(t0=a.t0, t1=a.t0 + (FS[-1] + 10) / a.fps, frames=len(FS), fps_analise=a.fps / 10),
               campo=dict(L=L, W=W), calibracao=dict(frames_afinados=sum(v['ok'] for v in FR.values()), frames=len(FR)),
               sentido={e: ('ataca para a direita' if SENT[e] > 0 else 'ataca para a esquerda') for e in EQ},
               visiveis={e: dict(mediana=float(np.median([sum(q[0] == e for q in J[f]) for f in FS])),
                                 frames_com_min=len(serie[e])) for e in EQ},
               min_vis=a.min_vis, formacao=formacao, resumos=resumos, series=serie,
               calor=dict(bins_x=bx.tolist(), bins_y=by.tolist(), **calor),
               bola=dict(frames_com_bola=len(B), territorio=territorio, posse=posse_pct, posse_frames=len(pv)),
               posicoes={str(f): dict(j=[[e, round(x, 1), round(y, 1)] for e, x, y, _ in J[f]],
                                      b=[round(B[f][0], 1), round(B[f][1], 1)] if f in B else None) for f in FS}),
          open(a.out, 'w'), ensure_ascii=False)
print('sentido:', {e: SENT[e] for e in EQ}, '| formação:', {e: formacao[e]['esquema'] for e in EQ})
print('posse:', posse_pct, '| território:', territorio, '| frames com bola:', len(B))
for e in EQ: print(e, {k: v['todos'] for k, v in resumos[e].items()}, 'frames:', len(serie[e]))
