"""Utilitários de campo: máscara de relva, máscara de linhas brancas, modelo do campo de 11."""
import cv2, numpy as np

def grass_mask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); H, S, V = cv2.split(hsv)
    g = ((H >= 30) & (H <= 90) & (S > 40) & (V > 40)).astype(np.uint8) * 255
    # fecha as linhas (finas) mas não os pilares/pessoas (largos); fica só a maior região
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    if n > 1: g = (np.isin(lab, 1 + np.where(st[1:, cv2.CC_STAT_AREA] > 0.02 * g.size)[0]) * 255).astype(np.uint8)
    return cv2.erode(g, np.ones((5, 5), np.uint8))

def white_mask(im, grass=None, boxes=()):
    """Linhas brancas: claras, pouco saturadas, mais claras que a vizinhança (top-hat), dentro da relva."""
    if grass is None: grass = grass_mask(im)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); S, V = hsv[..., 1], hsv[..., 2]
    th = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    w = ((S < 60) & (V > 150) & (th > 30) & (grass > 0)).astype(np.uint8) * 255
    for b in boxes:
        x1, y1, x2, y2 = map(int, b); w[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
    return w

def template(L=105.0, W=68.0, step=0.25):
    """Pontos densos das linhas do campo de 11 (metros; origem no centro; x ao comprimento, y à largura)."""
    P = []
    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step)); t = np.linspace(0, 1, n)[:, None]
        P.append(np.array(a) * (1 - t) + np.array(b) * t)
    def arc(c, r, a0, a1):
        n = max(8, int(r * abs(a1 - a0) / step)); a = np.linspace(a0, a1, n)
        P.append(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
    l, w = L / 2, W / 2
    seg((-l, -w), (l, -w)); seg((-l, w), (l, w)); seg((-l, -w), (-l, w)); seg((l, -w), (l, w)); seg((0, -w), (0, w))
    arc((0, 0), 9.15, 0, 2 * np.pi)
    for s in (-1, 1):
        bx = s * (l - 16.5); seg((s * l, -20.16), (bx, -20.16)); seg((s * l, 20.16), (bx, 20.16)); seg((bx, -20.16), (bx, 20.16))
        gx = s * (l - 5.5); seg((s * l, -9.16), (gx, -9.16)); seg((s * l, 9.16), (gx, 9.16)); seg((gx, -9.16), (gx, 9.16))
        a = np.arccos(5.5 / 9.15); c = (s * (l - 11), 0)
        arc(c, 9.15, -a, a) if s < 0 else arc(c, 9.15, np.pi - a, np.pi + a)
    return np.vstack(P)

def grass_top(g, run=15):
    """Limite superior da relva por coluna: primeira linha a partir da qual há relva contínua durante `run` px
    (ignora copas de árvores e manchas verdes soltas). NaN onde a coluna não tem relva (ex.: atrás de um pilar)."""
    solid = cv2.erode(g, np.ones((run, 1), np.uint8), anchor=(0, 0))
    has = solid.any(0)
    top = np.argmax(solid > 0, 0).astype(float); top[~has] = np.nan
    return top

def off_grass(boxes, top, margin=0.0, pad=30):
    """True para cada caixa cujos pés estão acima do limite da relva (banco, vedação, bancada do fundo).
    margin em alturas de pessoa (positivo = mais exigente). Sem relva visível à volta -> não filtra."""
    out = []
    for x1, y1, x2, y2 in boxes:
        seg = top[max(0, int(x1) - pad):min(len(top), int(x2) + pad)]
        seg = seg[~np.isnan(seg)]
        out.append(bool(len(seg) and (y2 - np.median(seg)) / max(1.0, y2 - y1) < margin))
    return out

def feet_on_grass(boxes, g, min_frac=0.3):
    """True se os pés da caixa estão em cima do relvado: fração de relva numa faixa logo abaixo dos pés
    (de y2 a y2 + 15 % da altura, 25 % mais larga que a caixa de cada lado; não inclui as pernas).
    Caixas cortadas pelo fundo da imagem -> False (pés invisíveis: público na bancada em primeiro plano)."""
    H, W = g.shape; out = []
    for x1, y1, x2, y2 in boxes:
        h, w = y2 - y1, x2 - x1
        if y2 >= H - 3: out.append(False); continue
        ya, yb = int(y2), int(min(H, y2 + .15 * h + 3))
        xa, xb = int(max(0, x1 - .25 * w)), int(min(W, x2 + .25 * w + 1))
        win = g[ya:yb, xa:xb]
        out.append(bool(win.size and (win > 0).mean() >= min_frac))
    return out
