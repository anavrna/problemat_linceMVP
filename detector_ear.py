"""
Teste básico de detecção de sonolência: EAR + PERCLOS com webcam.
Usa o MediaPipe Face Landmarker para achar os pontos do olho.

Teclas:  q = sair   |   c = recalibrar (olhe para a câmera com os olhos abertos)
"""
import os
import time
import threading
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------- Parâmetros (ajuste à vontade) -------------------------
FPS_ALVO       = 30     # requisito do projeto
JANELA_S       = 10     # janela do PERCLOS em segundos (curta para testar; o "clássico" é 60 s)
LIMIAR_PERCLOS = 0.30   # fração da janela com olho fechado que dispara o alarme
FATOR_FECHADO  = 0.70   # olho "fechado" se EAR < FATOR_FECHADO * EAR_base (calibrado)
CALIB_S        = 3.0    # segundos de calibração com os olhos abertos
INTERVALO_BIPE = 1.0    # segundos entre bipes enquanto o alarme estiver ativo

MODELO = "face_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")

# Índices da malha do MediaPipe, na ordem p1..p6 da fórmula do EAR
# p1 e p4 = cantos do olho; p2, p3 = pálpebra de cima; p6, p5 = pálpebra de baixo
OLHO_DIR = [33, 160, 158, 133, 153, 144]
OLHO_ESQ = [362, 385, 387, 263, 373, 380]


def calcula_ear(p):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = p
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal)


class BufferPerclos:
    """Buffer circular de 1 byte por quadro com soma acumulada.
    Atualização O(1) por quadro: é a mesma estrutura que iria para o firmware."""

    def __init__(self, n):
        self.n = n
        self.zerar()

    def zerar(self):
        self.buf = bytearray(self.n)
        self.i = 0
        self.soma = 0
        self.cheio = False

    def push(self, fechado):
        self.soma += int(fechado) - self.buf[self.i]   # entra o novo, sai o mais antigo
        self.buf[self.i] = int(fechado)
        self.i = (self.i + 1) % self.n
        if self.i == 0:
            self.cheio = True

    def valor(self):
        total = self.n if self.cheio else max(self.i, 1)
        return self.soma / total


def bipe():
    try:
        import winsound                      # Windows
        winsound.Beep(1000, 400)
    except ImportError:
        import subprocess, sys
        if sys.platform == "darwin":         # macOS: toca um som do sistema
            subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"])
        else:
            print("\a", end="", flush=True)  # Linux: sino do terminal


def main():
    if not os.path.exists(MODELO):
        print("Baixando o modelo do MediaPipe (~4 MB)...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)

    opcoes = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(opcoes)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_ALVO)

    buffer = BufferPerclos(int(JANELA_S * FPS_ALVO))
    ear_base, amostras_calib, t_calib = None, [], time.time()
    ultimo_bipe = 0.0
    tempos_ms = []                 # tempo de processamento por quadro
    t_inicio = time.perf_counter()
    n_quadros, t_fps, fps_real = 0, time.time(), 0.0

    print("Calibrando: olhe para a câmera com os olhos abertos...")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Não consegui ler a webcam.")
            break
        t_cap = time.perf_counter()          # instante em que o quadro chegou

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((t_cap - t_inicio) * 1000)
        res = detector.detect_for_video(img, ts_ms)

        status, cor = "SEM ROSTO", (0, 165, 255)
        ear = None

        if res.face_landmarks:
            lm = res.face_landmarks[0]
            pts = lambda idx: np.array([[lm[k].x * w, lm[k].y * h] for k in idx])
            p_dir, p_esq = pts(OLHO_DIR), pts(OLHO_ESQ)
            ear = (calcula_ear(p_dir) + calcula_ear(p_esq)) / 2.0

            for x, y in np.vstack([p_dir, p_esq]).astype(int):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear_base is None:
                # ---- fase de calibração ----
                amostras_calib.append(ear)
                status, cor = "CALIBRANDO (olhos abertos)", (255, 255, 0)
                if time.time() - t_calib >= CALIB_S:
                    ear_base = float(np.median(amostras_calib))
                    print(f"EAR base = {ear_base:.3f} | limiar fechado = {FATOR_FECHADO * ear_base:.3f}")
            else:
                # ---- operação normal ----
                fechado = ear < FATOR_FECHADO * ear_base
                buffer.push(fechado)
                status, cor = ("FECHADO", (0, 0, 255)) if fechado else ("ABERTO", (0, 255, 0))

        proc_ms = (time.perf_counter() - t_cap) * 1000
        tempos_ms.append(proc_ms)
        tempos_ms = tempos_ms[-100:]

        # ---- decisão: alarme por PERCLOS ----
        perclos = buffer.valor()
        alarme = ear_base is not None and buffer.cheio and perclos >= LIMIAR_PERCLOS
        if alarme and time.time() - ultimo_bipe >= INTERVALO_BIPE:
            ultimo_bipe = time.time()
            latencia_ms = (time.perf_counter() - t_cap) * 1000
            threading.Thread(target=bipe, daemon=True).start()
            print(f"ALERTA  PERCLOS={perclos:.0%}  latência quadro->alarme={latencia_ms:.1f} ms")

        # ---- FPS real da câmera ----
        n_quadros += 1
        if time.time() - t_fps >= 1.0:
            fps_real = n_quadros / (time.time() - t_fps)
            n_quadros, t_fps = 0, time.time()

        # ---- sobreposição na tela ----
        linhas = [
            f"Olho: {status}",
            f"EAR: {ear:.3f}" if ear is not None else "EAR: -",
            f"PERCLOS ({JANELA_S}s): {perclos:.0%}" + ("" if buffer.cheio else " (enchendo janela)"),
            f"Proc: {np.mean(tempos_ms):.1f} ms/quadro | FPS camera: {fps_real:.1f}",
        ]
        for k, txt in enumerate(linhas):
            cv2.putText(frame, txt, (10, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        if alarme:
            cv2.putText(frame, "SONOLENCIA!", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

        cv2.imshow("Teste EAR + PERCLOS", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            ear_base, amostras_calib, t_calib = None, [], time.time()
            buffer.zerar()
            print("Recalibrando...")

    cap.release()
    cv2.destroyAllWindows()
    if tempos_ms:
        print(f"Tempo médio de processamento: {np.mean(tempos_ms):.1f} ms/quadro")


if __name__ == "__main__":
    main()