import cv2
import numpy as np

BILD_PFAD = 'szene.png'
LABELS = ['oben-links', 'oben-rechts', 'unten-rechts', 'unten-links']
FARBEN = [(60, 60, 220), (50, 200, 50), (220, 130, 30), (30, 160, 220)]

bild = cv2.imread(BILD_PFAD)
anzeige = bild.copy()
geklickte_punkte = []


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(geklickte_punkte) >= 4:
        return

    idx = len(geklickte_punkte)
    geklickte_punkte.append([x, y])

    cv2.circle(anzeige, (x, y), 8, FARBEN[idx], -1)
    cv2.putText(anzeige, f'{LABELS[idx]} ({x},{y})',
                (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, FARBEN[idx], 2)

    if len(geklickte_punkte) == 4:
        pts = np.array(geklickte_punkte, dtype=np.int32)
        cv2.polylines(anzeige, [pts], isClosed=True,
                      color=(0, 220, 220), thickness=2)

    cv2.imshow('Eckpunkte setzen', anzeige)

    if len(geklickte_punkte) == 4:
        print('\ndst_pts = np.float32([')
        for pt in geklickte_punkte:
            print(f'    {pt},')
        print('])')
        print('\nFenster schliessen mit beliebiger Taste.')


cv2.namedWindow('Eckpunkte setzen')
cv2.setMouseCallback('Eckpunkte setzen', on_click)
cv2.imshow('Eckpunkte setzen', anzeige)
cv2.waitKey(0)  # "q" drücken um das Fenster zu schliessen
cv2.destroyAllWindows()
