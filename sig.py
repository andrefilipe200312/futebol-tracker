"""Assinatura de cor do tronco (partilhada pela deteção e pela visualização ao vivo)."""
import cv2, numpy as np

def torso_sig(im, b):
    """Histograma 12 tons x 3 saturações (píxeis coloridos) + 3 níveis (escuro/médio/claro) dos píxeis
    pouco saturados. A relva (verde) é excluída. Normalizado pela área útil."""
    x1, y1, x2, y2 = map(int, b); w, h = x2 - x1, y2 - y1
    c = im[max(0, y1 + int(.18 * h)):max(0, y1 + int(.55 * h)), max(0, x1 + int(.2 * w)):max(0, x2 - int(.2 * w))]
    if c.size < 12: return None, 0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int); H, S, V = hsv.T
    grass = (H >= 35) & (H <= 85) & (S > 50) & (V > 40)
    H, S, V = H[~grass], S[~grass], V[~grass]
    n = len(H)
    if n < 8: return None, n
    col = S >= 60
    hb = (H[col] * 12 // 180).clip(0, 11); sb = ((S[col] - 60) * 3 // 196).clip(0, 2)
    hist = np.bincount(hb * 3 + sb, minlength=36).astype(float)
    vb = (V[~col] * 3 // 256).clip(0, 2); g = np.bincount(vb, minlength=3).astype(float)
    sig = np.r_[hist, g] / n
    return [round(float(v), 4) for v in sig], int(n)
