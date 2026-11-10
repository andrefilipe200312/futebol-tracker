"""Pipeline completo: deteção -> equipa (cor) -> filtros (relvado, banco, GR) -> seguimento entre frames.
Funciona localmente (OpenVINO no CPU) e no Colab (GPU/CUDA, escolhido automaticamente).
Processa por trechos (--chunk segundos); cada trecho fica em <saida>/chunks/ e é saltado se já existir,
por isso pode ser interrompido e retomado. No fim junta tudo em <saida>/jogadores.csv.

Uso: python process.py <video> <saida> [--fps 5] [--t0 0] [--t1 fim] [--chunk 300] [--video] [--show]
  --video  grava um mp4 por trecho com caixas, IDs e equipa
  --show   mostra uma janela ao vivo (só localmente)"""
import os, sys, json, csv, time, argparse
import cv2, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sig import torso_sig
from teams import label_frame, draw, bench_filter, COL
from tracker import Tracker, camera_motion, GR_SHARE, TEAMS

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('out')
ap.add_argument('--fps', type=float, default=5); ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=None)
ap.add_argument('--chunk', type=float, default=300); ap.add_argument('--reg-every', type=float, default=1.0, help='segundos entre registos completos nas referências')
ap.add_argument('--video', dest='save_video', action='store_true'); ap.add_argument('--show', action='store_true')
ap.add_argument('--teams-model', default=str(HERE / 'teams_model.npz'))
ap.add_argument('--labels', default='0=rosa,1=rosa,4=vermelho,5=arbitro,2=fora,3=fora')
a = ap.parse_args()

out = Path(a.out); (out / 'chunks').mkdir(parents=True, exist_ok=True)
C = np.load(a.teams_model)['C']; lab_of = dict(x.split('=') for x in a.labels.split(','))

def load_model():
    from ultralytics import YOLO
    try:
        import torch
        if torch.cuda.is_available():
            print('deteção: GPU', torch.cuda.get_device_name(0), flush=True)
            return YOLO(str(HERE / 'yolov8n.pt')), dict(device=0, half=True)
    except Exception: pass
    if (HERE / 'ov1280_openvino_model').exists():
        print('deteção: OpenVINO (CPU)', flush=True)
        return YOLO(str(HERE / 'ov1280_openvino_model'), task='detect'), {}
    print('deteção: PyTorch (CPU)', flush=True)
    return YOLO(str(HERE / 'yolov8n.pt')), {}
model, kw = load_model()
bf = bench_filter()

cap = cv2.VideoCapture(a.video); vfps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = N / vfps; t1 = min(a.t1 or dur, dur)
step = vfps / a.fps; reg_every = max(1, int(round(a.reg_every * a.fps)))
print(f'vídeo {dur:.0f} s @ {vfps:.0f} fps; amostra {a.fps} fps; trechos de {a.chunk:.0f} s', flush=True)

def online_label(t):
    """Rótulo provisório (para o vídeo): maioria até agora, GR se já foi GR em >= GR_SHARE."""
    n = sum(t.labels.values())
    if t.labels['gr'] >= GR_SHARE * n: return 'gr'
    return max(t.labels.items(), key=lambda x: x[1])[0]

def run_chunk(c0, c1, name):
    global STOP
    tr = Tracker(a.fps); rows = []; pos = None
    vw = cv2.VideoWriter(str(out / 'chunks' / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (1280, 720)) if a.save_video else None
    prev_gray = prev_boxes = None; regs = []; k = 0; tstart = time.time()
    first = int(round(c0 * vfps)); cap.set(cv2.CAP_PROP_POS_FRAMES, first); pos = first
    while True:
        f = int(round(first + k * step))
        if f / vfps >= c1 or f >= N: break
        while pos < f: cap.grab(); pos += 1
        ok, im = cap.read(); pos += 1
        if not ok: break
        r = model.predict(im, imgsz=[736, 1280], conf=0.25, classes=[0], verbose=False, **kw)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist(); confs = r.boxes.conf.cpu().numpy().tolist()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        G = camera_motion(prev_gray, gray, prev_boxes or [])
        # registo nas referências: completo a cada reg_every frames (ou se o movimento da câmara falhar), senão propagado
        if k % reg_every == 0 or G is None: regs = bf.register(im, boxes)
        else: regs = [(H @ np.linalg.inv(G), ref, n) for H, ref, n in regs]
        zones = bf.zones(boxes, regs)
        lab = label_frame(im, [(b, torso_sig(im, b)[0]) for b in boxes], C, lab_of, zones=zones)
        tracked = tr.step(k, [(b, l) for l, b in lab], G)
        tmap = {t.id: t for t in tr.tracks}
        for tid, b, l in tracked:
            rows.append(dict(t=round(f / vfps, 2), k=k, track=tid, equipa_frame=l, x_pes=round((b[0] + b[2]) / 2, 1), y_pes=round(b[3], 1),
                             x1=round(b[0], 1), y1=round(b[1], 1), x2=round(b[2], 1), y2=round(b[3], 1)))
        if vw is not None or a.show:
            vis = draw(im, [(l, b) for l, b in lab if l not in ('vermelho', 'rosa', 'arbitro', 'gr')], f / vfps)
            cnt = {}
            for tid, b, l in tracked:
                ol = online_label(tmap[tid]) if tid in tmap else l; cnt[ol] = cnt.get(ol, 0) + 1
                x1, y1, x2, y2 = map(int, b); cv2.rectangle(vis, (x1, y1), (x2, y2), COL.get(ol, (200, 200, 200)), 2)
                cv2.putText(vis, str(tid), (x1, max(45, y1 - 4)), 0, .5, COL.get(ol, (200, 200, 200)), 2)
            cv2.rectangle(vis, (0, 0), (1280, 34), (25, 25, 25), -1)
            cv2.putText(vis, f"t={f/vfps:.1f}s  vermelho {cnt.get('vermelho',0)}  rosa {cnt.get('rosa',0)}  arbitro {cnt.get('arbitro',0)}  GR {cnt.get('gr',0)}  (seguidos: {len(tr.tracks)})",
                        (12, 24), 0, .65, (255, 255, 255), 2)
            if vw is not None: vw.write(vis)
            if a.show:
                cv2.imshow('equipas', vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    STOP = True; print('parado pelo utilizador (q/Esc); a gravar o que foi feito', flush=True); break
        prev_gray, prev_boxes = gray, boxes; k += 1
        if k % 100 == 0: print(f'  {name}: t={f/vfps:.0f}s {(time.time()-tstart)/k:.2f}s/frame, trajetórias ativas {len(tr.tracks)}', flush=True)
    if a.show: cv2.destroyAllWindows()
    if vw is not None: vw.release()
    final = tr.final_labels()
    with open(out / 'chunks' / f'{name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['t', 'k', 'track', 'equipa', 'equipa_frame', 'x_pes', 'y_pes', 'x1', 'y1', 'x2', 'y2']); w.writeheader()
        for r in rows:
            fl = final.get(r['track'])
            if fl is None: continue
            w.writerow(dict(r, track=f"{name}_{r['track']}", equipa=fl))
    n = sum(1 for v in final.values() if v); print(f'{name}: {k} frames, {n} trajetórias, {(time.time()-tstart)/max(1,k):.2f}s/frame', flush=True)

STOP = False
c0 = a.t0
while c0 < t1 and not STOP:
    c1 = min(c0 + a.chunk, t1); name = f'trecho_{int(c0):05d}_{int(c1):05d}'
    if (out / 'chunks' / f'{name}.csv').exists(): print(name, 'já feito', flush=True)
    else: run_chunk(c0, c1, name)
    c0 = c1
# junta tudo
parts = sorted((out / 'chunks').glob('trecho_*.csv'))
with open(out / 'jogadores.csv', 'w', newline='') as fo:
    for i, p in enumerate(parts):
        lines = open(p).read().splitlines()
        fo.write('\n'.join(lines if i == 0 else lines[1:]) + '\n')
print('FEITO ->', out / 'jogadores.csv', flush=True)
