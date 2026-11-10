"""Avalia cada configuração contra a revisão manual (dados/comparacao/revisao.json).
Uma deteção conta como bola se estiver a <= 10 px de um candidato marcado 'bola' no mesmo frame;
deteções perto de candidatos 'duvida' não contam nem a favor nem contra. Tudo o resto é falso positivo.
Para cada configuração: revocação e falsos por frame a vários limiares de confiança, e o melhor F1.
Uso: python avaliar.py <pasta_comparacao>"""
import sys, json
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
C = {c['id']: c for c in json.load(open(d / 'candidatos.json'))}; R = json.load(open(d / 'revisao.json'))
B = [C[i] for i in R['bola']]; jogo = [C[i] for i in R['bola'] if i not in R['suplente']]; Q = [C[i] for i in R['duvida']]
def near(f, x, y, L): return [c for c in L if c['frame'] == f and np.hypot(c['x'] - x, c['y'] - y) <= 10]
rows = []
for p in sorted(d.glob('dets_*.json')):
    J = json.load(open(p)); nf = len(J['dets']); best = None; at = {}
    for th in [0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        hit, hitj, fp = set(), set(), 0
        for f, L in J['dets'].items():
            for x in L:
                if x['conf'] < th: continue
                b = x['xyxy']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                m = near(int(f), cx, cy, B)
                if m: hit.update(id(c) for c in m); hitj.update(id(c) for c in m if c in jogo)
                elif not near(int(f), cx, cy, Q): fp += 1
        rec = len(hit) / len(B); prec = len(hit) / max(1, len(hit) + fp); f1 = 2 * rec * prec / max(1e-9, rec + prec)
        at[th] = (rec, len(hitj) / len(jogo), fp / nf, prec, f1)
        if best is None or f1 > best[1]: best = (th, f1)
    rows.append((J['model'], J['imgsz'], J['s_por_frame'], at, best))
print(f"{len(B)} bolas confirmadas ({len(jogo)} em jogo, {len(B)-len(jogo)} suplentes paradas), {len(Q)} dúvidas, 60 frames\n")
print(f"{'config':16s} {'s/frame':>7s} | {'revoc@.05':>9s} {'jogo@.05':>8s} {'falsos/fr@.05':>13s} | {'revoc@.15':>9s} {'falsos/fr@.15':>13s} | melhor F1 (limiar)")
for m, sz, s, at, (th, f1) in sorted(rows, key=lambda r: -r[4][1]):
    a, b = at[0.05], at[0.15]
    print(f"{m+' @'+str(sz):16s} {s:7.2f} | {a[0]:9.0%} {a[1]:8.0%} {a[2]:13.2f} | {b[0]:9.0%} {b[2]:13.2f} | {f1:.2f} ({th})")
