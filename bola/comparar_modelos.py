"""Compara modelos pré-treinados (COCO, classe 32 'sports ball') na deteção da bola.
Corre cada configuração (modelo x tamanho de entrada) nos mesmos frames do trecho e guarda todas as
deteções de bola com conf >= 0.03 em <saida>/dets_<config>.json, mais o tempo por frame.
Uso: python comparar_modelos.py <video_trecho> <saida> [passo_frames=50]"""
import sys, json, time
from pathlib import Path
import cv2
from ultralytics import YOLO, RTDETR

HERE = Path(__file__).resolve().parent
video, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
PASSO = int(sys.argv[3]) if len(sys.argv) > 3 else 50
CONFIGS = [('yolov8n', 1280), ('yolov8n', 1920), ('yolo11n', 1280), ('yolo11n', 1920), ('yolo11s', 1280),
           ('yolo26n', 1280), ('yolo26n', 1920), ('yolo26s', 1280), ('yolo26s', 1920), ('yolo26m', 1280),
           ('rtdetr-l', 1280)]

cap = cv2.VideoCapture(video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = {}
for f in range(0, N, PASSO):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    if ok: frames[f] = im; cv2.imwrite(str(out / f'f{f:04d}.png'), im)
print(len(frames), 'frames', flush=True)

for name, sz in CONFIGS:
    dst = out / f'dets_{name}_{sz}.json'
    if dst.exists(): print(dst.name, 'já feito'); continue
    m = (RTDETR if 'rtdetr' in name else YOLO)(str(HERE / 'pesos' / f'{name}.pt'))
    m.predict(next(iter(frames.values())), imgsz=sz, verbose=False)
    res = {}; t = time.time()
    for f, im in frames.items():
        r = m.predict(im, imgsz=sz, classes=[32], conf=0.03, verbose=False)[0]
        res[f] = [dict(xyxy=[round(v, 1) for v in b], conf=round(c, 3)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    dt = (time.time() - t) / len(frames)
    json.dump(dict(model=name, imgsz=sz, s_por_frame=round(dt, 2), dets=res), open(dst, 'w'))
    print(f'{name} {sz}: {dt:.2f}s/frame, {sum(len(v) for v in res.values())} deteções', flush=True)
print('FEITO', flush=True)
