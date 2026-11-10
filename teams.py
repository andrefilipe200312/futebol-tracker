"""Separação de equipas por agrupamento de cor do tronco (k-means, k=6, distância de Hellinger).
Passo 1 (sem mapa): python teams.py fit <pasta>            -> teams_model.npz + mosaico por grupo (para rotular)
Passo 2:            python teams.py apply <pasta> <t_ini> <t_fim> <mapa grupo=rotulo ...>
                    -> teams.csv (posições por frame) e teams.mp4"""
import sys, json, csv, cv2, numpy as np, random
from pathlib import Path

from pitch import grass_mask, grass_top, off_grass, feet_on_grass

COL = {'vermelho': (40, 40, 230), 'rosa': (200, 120, 255), 'arbitro': (0, 230, 255), 'fora': (120, 120, 120), 'banco': (255, 160, 0), 'gr': (255, 255, 255)}

# Filtro do banco pelo limite da relva: DESLIGADO por omissão. No teste (0-60 s) tirava o banco mas também
# ~1/3 de jogadores reais junto à baliza do fundo (pés à mesma altura do limite da relva que o banco).
# Ligar com a variável de ambiente FILTRO_BANCO=1.
import os
BENCH_RULE = os.environ.get('FILTRO_BANCO') == '1'

GR_DEPTH = 0.4   # fração da profundidade da grande área (a partir da linha de golo) onde se aceita o GR. Teste: GR 0,10-0,44; árbitro num livre 0,46-0,49
_bench = None
def bench_filter():
    """Camada 2 (ligada): pés dentro de um abrigo marcado em zones.json -> 'banco'."""
    global _bench
    if _bench is None:
        from bench import BenchFilter
        _bench = BenchFilter(str(Path(__file__).parent / 'zones.json'))
    return _bench

def label_frame(im, dets, C, lab_of, zones=None):
    """dets: lista de (xyxy, assinatura). Pés num abrigo (camada 2) -> 'banco'; com BENCH_RULE também
    pés acima do limite da relva (camada 1). Senão, equipa pela cor."""
    boxes = [b for b, _ in dets]
    if not boxes: return []
    g = grass_mask(im)
    if zones is None: zones = bench_filter()(im, boxes)[0]   # (zona, profundidade) por caixa
    off = [bool(z and z[0].startswith('banco')) for z in zones]
    if BENCH_RULE: off = [a or b for a, b in zip(off, off_grass(boxes, grass_top(g)))]
    on = feet_on_grass(boxes, g)                     # só é jogador quem tem os pés no relvado
    out = [('banco' if o else classify(s, C, lab_of) if ok else 'fora', b) for (b, s), o, ok in zip(dets, off, on)]
    # guarda-redes: veste preto (grupo do árbitro, ou escuro) e está na metade da grande área do lado da baliza.
    # No máximo um por área: o mais perto da linha de golo (o árbitro nos lances parados fica mais à frente).
    best = {}
    for i, ((lab, b), z, ok) in enumerate(zip(out, zones, on)):
        if ok and lab in ('arbitro', 'fora') and z and z[0].startswith('area') and z[1] is not None and z[1] <= GR_DEPTH:
            if z[0] not in best or z[1] < best[z[0]][1]: best[z[0]] = (i, z[1])
    for i, _ in best.values(): out[i] = ('gr', out[i][1])
    return out

def classify(sig, C, lab_of):
    if not sig: return 'fora'
    return lab_of.get(str(int(np.argmin(((C - np.sqrt(np.float32(sig))) ** 2).sum(1)))), 'fora')

def draw(im, labelled, t):
    """labelled: lista de (equipa, xyxy). Desenha caixas e elipse nos pés, e a contagem no topo."""
    cnt = {}
    for team, (x1, y1, x2, y2) in labelled:
        cnt[team] = cnt.get(team, 0) + 1
        if team in ('fora', 'banco'): cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), COL[team], 1); continue
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), COL[team], 2)
        cv2.ellipse(im, (int((x1 + x2) / 2), int(y2)), (max(6, int((x2 - x1) / 2)), 4), 0, 0, 360, COL[team], 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (25, 25, 25), -1)
    txt = f"t={t:.1f}s   vermelho {cnt.get('vermelho', 0)}   rosa {cnt.get('rosa', 0)}   arbitro {cnt.get('arbitro', 0)}   GR {cnt.get('gr', 0)}   banco {cnt.get('banco', 0)}   fora {cnt.get('fora', 0)}"
    cv2.putText(im, txt, (12, 24), 0, .7, (255, 255, 255), 2)
    return im

def load(root): return sorted([json.loads(l) for l in open(Path(root) / 'detections.jsonl')], key=lambda d: d['k'])

if __name__ == '__main__' and sys.argv[1] == 'fit':
    root = Path(sys.argv[2]); D = load(root); X = []; ref = []
    for f in D:
        for d in f['dets']:
            if d['sig'] and d['conf'] > .25: X.append(d['sig']); ref.append((f['frame'], d['xyxy']))
    X = np.sqrt(np.float32(X)); cv2.setRNGSeed(0)
    _, lab, C = cv2.kmeans(X, 6, None, (3, 100, 1e-4), 8, cv2.KMEANS_PP_CENTERS); lab = lab.ravel()
    np.savez(root / 'teams_model.npz', C=C)
    random.seed(0); rows = []
    for k in range(6):
        idx = [i for i in range(len(X)) if lab[i] == k]; cr = []
        for i in random.sample(idx, min(24, len(idx))):
            im = cv2.imread(str(root / 'frames' / ref[i][0])); x1, y1, x2, y2 = map(int, ref[i][1])
            cr.append(cv2.resize(im[max(0, y1):y2, max(0, x1):x2], (40, 80)))
        cr += [np.zeros((80, 40, 3), np.uint8)] * (24 - len(cr))
        row = cv2.hconcat(cr); cv2.putText(row, f'{k} (n={len(idx)})', (2, 20), 0, .6, (0, 255, 255), 2); rows.append(row)
    cv2.imwrite(str(root / 'teams_groups.jpg'), cv2.vconcat(rows)); print('grupos em', root / 'teams_groups.jpg')
elif __name__ == '__main__':
    root = Path(sys.argv[2]); t0, t1 = float(sys.argv[3]), float(sys.argv[4])
    lab_of = dict(a.split('=') for a in sys.argv[5:])
    C = np.load(root / 'teams_model.npz')['C']
    D = [f for f in load(root) if t0 <= f['t'] < t1]
    fps = round(1 / (D[1]['t'] - D[0]['t'])) if len(D) > 1 else 1
    vw = cv2.VideoWriter(str(root / 'teams.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
    cw = csv.writer(open(root / 'teams.csv', 'w', newline='')); cw.writerow(['t', 'frame', 'equipa', 'x_pes', 'y_pes', 'x1', 'y1', 'x2', 'y2', 'conf'])
    for f in D:
        im = cv2.imread(str(root / 'frames' / f['frame']))
        ds = [d for d in f['dets'] if d['conf'] >= .25]
        lab = label_frame(im, [(d['xyxy'], d['sig']) for d in ds], C, lab_of)
        for (team, (x1, y1, x2, y2)), d in zip(lab, ds):
            cw.writerow([f['t'], f['frame'], team, round((x1 + x2) / 2, 1), y2, x1, y1, x2, y2, d['conf']])
        vw.write(draw(im, lab, f['t']))
    vw.release(); print(len(D), 'frames ->', root / 'teams.mp4', root / 'teams.csv')
