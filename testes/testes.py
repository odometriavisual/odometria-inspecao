import cv2
import numpy as np
from tkinter import Tk, filedialog

# =========================
# SELEÇÃO DO VÍDEO
# =========================
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Selecione o vídeo",
    filetypes=[("Vídeos", "*.mp4 *.avi *.mov")]
)

if not video_path:
    raise SystemExit("Nenhum vídeo selecionado.")

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

if not ret:
    raise SystemExit("Erro ao ler o vídeo.")

# =========================
# SELEÇÃO DA ROI (CANO)
# =========================
roi = cv2.selectROI(
    "Selecione o CANO (pressione ENTER)",
    frame,
    showCrosshair=True,
    fromCenter=False
)
cv2.destroyWindow("Selecione o CANO (pressione ENTER)")

x, y, w, h = roi
if w == 0 or h == 0:
    raise SystemExit("ROI inválida.")

roi_bgr = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

# =========================
# ESTATÍSTICAS DE COR
# =========================
h_vals = roi_hsv[:, :, 0].reshape(-1)
s_vals = roi_hsv[:, :, 1].reshape(-1)
v_vals = roi_hsv[:, :, 2].reshape(-1)

h_mean, h_std = np.mean(h_vals), np.std(h_vals)
s_mean, s_std = np.mean(s_vals), np.std(s_vals)
v_mean, v_std = np.mean(v_vals), np.std(v_vals)

# fator de tolerância (ajuste fino)
K = 2.5

lower = np.array([
    max(0,   h_mean - K * h_std),
    max(0,   s_mean - K * s_std),
    max(0,   v_mean - K * v_std)
], dtype=np.uint8)

upper = np.array([
    min(180, h_mean + K * h_std),
    min(255, s_mean + K * s_std),
    min(255, v_mean + K * v_std)
], dtype=np.uint8)

print("=== INTERVALO HSV CALIBRADO ===")
print("Lower:", lower)
print("Upper:", upper)

# =========================
# PREVIEW DA MÁSCARA
# =========================
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

overlay = frame.copy()
overlay[mask > 0] = (0.5 * overlay[mask > 0] + np.array([0, 255, 0]) * 0.5)

cv2.imshow("Cano detectado (verde)", overlay)
cv2.imshow("Mascara do cano", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
