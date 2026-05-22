"""
Gera folha com marcadores ArUco IDs 0-7 (DICT_6X6_250)
para usar com a Ocarina Virtual do TP2.

  ID 0 = corpo da ocarina
  ID 1 = Dó   ID 2 = Ré   ID 3 = Mi   ID 4 = Fá
  ID 5 = Sol  ID 6 = Lá   ID 7 = Si

Uso:
    python gera_aruco.py
    → salva aruco_ocarina_ids_0a7.png  (imprima em tamanho real ~5 cm cada)
"""

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

NOTE_NAMES = {0:"CORPO", 1:"Dó", 2:"Ré", 3:"Mi", 4:"Fá", 5:"Sol", 6:"Lá", 7:"Si"}

MARKER_PX  = 200   # tamanho de cada marcador em pixels
PADDING    = 30    # espaço entre marcadores
COLS       = 4
ROWS       = 2     # 8 marcadores → 2 linhas de 4

CELL       = MARKER_PX + PADDING
IMG_W      = COLS * CELL + PADDING
IMG_H      = ROWS * CELL + PADDING + 30   # +30 para legenda no rodapé

img = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255

for i in range(8):
    col = i % COLS
    row = i // COLS
    x = PADDING + col * CELL
    y = PADDING + row * CELL

    marker = cv2.aruco.generateImageMarker(ARUCO_DICT, i, MARKER_PX)
    img[y:y+MARKER_PX, x:x+MARKER_PX] = marker

    label = f"ID:{i}  {NOTE_NAMES[i]}"
    cv2.putText(img, label, (x, y + MARKER_PX + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)

# Rodapé
cv2.putText(img, "DICT_6X6_250  |  Imprima com cada marcador ~5 cm  |  TP2 Visao Computacional",
            (PADDING, IMG_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, 80, 1, cv2.LINE_AA)

out_path = "aruco_ocarina_ids_0a7.png"
cv2.imwrite(out_path, img)
print(f"[OK] Salvo: {out_path}")
print("Imprima e recorte cada marcador com ~5 cm de lado.")