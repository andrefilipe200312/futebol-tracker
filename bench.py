"""Camada 2 do filtro do banco: o banco é um lugar fixo do estádio.
A câmara está sempre no mesmo sítio e só roda/faz zoom, por isso cada frame liga-se a uma imagem de
referência por uma homografia. Usa-se o fundo (prédios, abrigos, vedação), que tem textura, e não a relva.
Os abrigos são marcados uma vez na referência (retângulos onde ficam os pés de quem lá está).
Uma deteção com os pés dentro de um abrigo -> 'banco'. Se o registo falhar, a regra não se aplica.

Também serve para outras zonas fixas (ex.: grandes áreas, para os guarda-redes).
Configuração em zones.json:
  {"referencias": [{"frame": "<jpg>", "caixas": [...], "zonas": {"banco_A": [x1,y1,x2,y2], "area_dir": [[x,y],...]}}]}
Cada referência que registar com o frame contribui com as suas zonas."""
import json, cv2, numpy as np
from pathlib import Path
from pitch import grass_mask, grass_top

MIN_INLIERS = 25
sift = cv2.SIFT_create(3000)
matcher = cv2.BFMatcher()

def background_mask(im, boxes, below=25):
    """Zona acima da relva (+ uma faixa de `below` px para incluir a base dos abrigos), sem pessoas."""
    h, w = im.shape[:2]
    top = grass_top(grass_mask(im))
    t = np.where(np.isnan(top), h * 0.5, top)
    t = cv2.blur(t.reshape(1, -1).astype(np.float32), (1, 61)).ravel()
    m = (np.arange(h)[:, None] < t[None, :] + below).astype(np.uint8) * 255
    m[:int(h * 0.04)] = 0                                   # moldura/estrutura do topo
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1) - 3):int(y2) + 3, max(0, int(x1) - 3):int(x2) + 3] = 0
    return m

def features(im, boxes):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, background_mask(im, boxes))

class BenchFilter:
    def __init__(self, config):
        cfg = json.load(open(config)); base = Path(config).parent
        self.refs = []
        for r in cfg['referencias']:
            im = cv2.imread(str(base / r['frame']))
            boxes = r.get('caixas', [])
            kp, des = features(im, boxes)
            zones = {k: np.array(v, float) for k, v in r['zonas'].items()}
            # áreas: transformação imagem -> quadrado unitário (x = largura, y = profundidade a partir da linha de golo)
            sq = {k: cv2.getPerspectiveTransform(np.float32(z), np.float32([[0, 0], [0, 1], [1, 1], [1, 0]]))
                  for k, z in zones.items() if k.startswith('area') and z.shape == (4, 2)}
            self.refs.append(dict(kp=kp, des=des, zones=zones, sq=sq))

    def register(self, im, boxes):
        """Homografias frame -> cada referência que registe bem. Devolve lista de (H, ref, n_inliers)."""
        kp, des = features(im, boxes)
        if des is None or len(kp) < MIN_INLIERS: return []
        out = []
        for ref in self.refs:
            if ref['des'] is None: continue
            m = matcher.knnMatch(des, ref['des'], k=2)
            good = [a for a, *b in m if b and a.distance < 0.75 * b[0].distance]
            if len(good) < MIN_INLIERS: continue
            p = np.float32([kp[g.queryIdx].pt for g in good]); q = np.float32([ref['kp'][g.trainIdx].pt for g in good])
            H, inl = cv2.findHomography(p, q, cv2.RANSAC, 3.0)
            if H is None: continue
            n = int(inl.sum()); s = np.linalg.det(H[:2, :2])
            # sanidade: câmara que só roda/faz zoom -> escala entre 1/5 e 5, sem espelhamento
            if n >= MIN_INLIERS and 0.04 < s < 25: out.append((H, ref, n))
        return out

    def __call__(self, im, boxes):
        """Para cada caixa: (nome da zona, profundidade) onde estão os pés, ou None; e o maior nº de inliers.
        A profundidade só existe nas áreas (0 = linha de golo, 1 = frente da área); nas outras zonas é None."""
        regs = self.register(im, boxes) if len(boxes) else []
        return self.zones(boxes, regs), max([r[2] for r in regs], default=0)

    def zones(self, boxes, regs):
        """Como __call__, mas com registos já calculados (ex.: propagados pelo movimento da câmara)."""
        res = [None] * len(boxes)
        if not len(boxes): return res
        feet = np.float32([[(b[0] + b[2]) / 2, b[3]] for b in boxes]).reshape(-1, 1, 2)
        for H, ref, n in sorted(regs, key=lambda r: -r[2]):
            fr = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
            for i, (x, y) in enumerate(fr):
                if res[i] is not None: continue
                k = next((k for k, z in ref['zones'].items() if inside(z, x, y)), None)
                if k is None: continue
                d = None
                if k in ref['sq']: d = float(cv2.perspectiveTransform(np.float32([[[x, y]]]), ref['sq'][k])[0, 0, 1])
                res[i] = (k, d)
        return res

def inside(z, x, y):
    if z.ndim == 1: return z[0] <= x <= z[2] and z[1] <= y <= z[3]
    return cv2.pointPolygonTest(z.astype(np.float32), (float(x), float(y)), False) >= 0
