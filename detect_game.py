"""Deteção de pessoas no jogo inteiro, amostrado.
Guarda: frames JPG (para calibração e vídeo), caixas YOLOv8 e uma assinatura de cor do tronco
(histograma tom x saturação + níveis de cinzento, sem os píxeis de relva).
Retoma onde parou se for interrompido.
Uso: python detect_game.py <video> <pasta_saida> [fps_amostra=1] [t_ini=0] [t_fim=fim] [max_frames]"""
import sys, json, time
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from sig import torso_sig

video, outdir = sys.argv[1], Path(sys.argv[2])
SFPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
T0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
T1 = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
MAXF = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
(outdir / 'frames').mkdir(parents=True, exist_ok=True)
detf = outdir / 'detections.jsonl'
done = set()
if detf.exists():
    for l in open(detf):
        try: done.add(json.loads(l)['k'])
        except Exception: pass

model = YOLO(str(Path(__file__).parent / 'ov1280_openvino_model'), task='detect')  # OpenVINO: 2x mais rápido que PyTorch neste CPU, mesmas deteções
cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = vfps / SFPS
targets = [int(round(k * step)) for k in range(int(N / step) + 1)]
targets = [(k, f) for k, f in enumerate(targets) if T0 <= f / vfps < T1 and k not in done][:MAXF]
print(f'video {N} frames @ {vfps} fps; amostra {SFPS} fps; por fazer {len(targets)}', flush=True)
if not targets: sys.exit()
first = targets[0][1]
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, first - 1))
pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
out = open(detf, 'a'); t_start = time.time()
for n_done, (k, f) in enumerate(targets):
    while pos < f: cap.grab(); pos += 1
    ok, im = cap.read(); pos += 1
    if not ok: break
    name = f'{k:06d}.jpg'; cv2.imwrite(str(outdir / 'frames' / name), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    r = model.predict(im, imgsz=[736, 1280], conf=0.15, classes=[0], verbose=False)[0]
    dets = []
    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        sig, npx = torso_sig(im, b)
        dets.append(dict(xyxy=[round(v, 1) for v in b], conf=round(c, 3), sig=sig, npx=npx))
    out.write(json.dumps(dict(k=k, vf=f, t=round(f / vfps, 3), frame=name, dets=dets)) + '\n'); out.flush()
    if n_done % 50 == 0:
        el = time.time() - t_start; print(f'{n_done+1}/{len(targets)} t={f/vfps:.0f}s dets={len(dets)} {el/(n_done+1):.2f}s/frame', flush=True)
print('DONE', flush=True)
