"""
Deteccao de sonolencia - teste com webcam
EAR (abertura do olho) + PERCLOS (porcentagem de tempo com o olho fechado)

Teclas: q = sair | c = calibrar de novo
"""

import os
import time
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


FPS = 30
JANELA_SEG = 10
LIMIAR_PERCLOS = 0.30
FATOR_FECHADO = 0.70
TEMPO_CALIB = 3
PAUSA_BIPE = 1

TAMANHO_JANELA = JANELA_SEG * FPS

ARQUIVO_MODELO = "face_landmarker.task"
LINK_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# pontos do contorno do olho na malha do MediaPipe
OLHO_DIREITO = [33, 160, 158, 133, 153, 144]
OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]


def calcular_ear(pontos):
    """Razao entre altura e largura do olho. Perto de zero = olho fechado."""
    p1, p2, p3, p4, p5, p6 = pontos
    altura = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    largura = np.linalg.norm(p1 - p4)
    return altura / (2 * largura)


def pegar_pontos(landmarks, indices, largura, altura):
    pontos = []
    for i in indices:
        x = landmarks[i].x * largura
        y = landmarks[i].y * altura
        pontos.append([x, y])
    return np.array(pontos)


def tocar_bipe():
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
    elif sys.platform == "win32":
        import winsound
        winsound.Beep(1000, 400)
    else:
        print("\a", end="", flush=True)


def baixar_modelo():
    if not os.path.exists(ARQUIVO_MODELO):
        print("Baixando o modelo do MediaPipe...")
        urllib.request.urlretrieve(LINK_MODELO, ARQUIVO_MODELO)
        print("Modelo baixado.")


def criar_detector():
    opcoes = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ARQUIVO_MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opcoes)


def main():
    baixar_modelo()
    detector = criar_detector()

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    janela = []
    ear_normal = None
    amostras_calib = []
    inicio_calib = time.time()
    ultimo_bipe = 0
    tempos = []

    inicio = time.time()
    contador_fps = 0
    marca_fps = time.time()
    fps_real = 0

    print("Calibrando. Olhe para a camera com os olhos abertos.")

    while True:
        deu_certo, quadro = camera.read()
        if not deu_certo:
            print("Nao consegui ler a webcam.")
            break

        comeco = time.time()
        quadro = cv2.flip(quadro, 1)
        altura, largura = quadro.shape[:2]

        rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect_for_video(imagem, int((comeco - inicio) * 1000))

        ear = None
        texto_olho = "sem rosto"
        cor = (0, 165, 255)

        if resultado.face_landmarks:
            marcos = resultado.face_landmarks[0]
            olho_dir = pegar_pontos(marcos, OLHO_DIREITO, largura, altura)
            olho_esq = pegar_pontos(marcos, OLHO_ESQUERDO, largura, altura)
            ear = (calcular_ear(olho_dir) + calcular_ear(olho_esq)) / 2

            for ponto in np.vstack([olho_dir, olho_esq]):
                cv2.circle(quadro, (int(ponto[0]), int(ponto[1])), 2, (0, 255, 0), -1)

            if ear_normal is None:
                amostras_calib.append(ear)
                texto_olho = "calibrando"
                cor = (255, 255, 0)
                if time.time() - inicio_calib >= TEMPO_CALIB:
                    ear_normal = float(np.median(amostras_calib))
                    print("EAR normal:", round(ear_normal, 3))
                    print("Limiar de olho fechado:", round(FATOR_FECHADO * ear_normal, 3))
            else:
                fechado = ear < FATOR_FECHADO * ear_normal
                janela.append(1 if fechado else 0)
                if len(janela) > TAMANHO_JANELA:
                    janela.pop(0)
                texto_olho = "fechado" if fechado else "aberto"
                cor = (0, 0, 255) if fechado else (0, 255, 0)

        tempos.append((time.time() - comeco) * 1000)
        if len(tempos) > 100:
            tempos.pop(0)

        if janela:
            perclos = sum(janela) / len(janela)
        else:
            perclos = 0

        janela_cheia = len(janela) == TAMANHO_JANELA
        alarme = janela_cheia and perclos >= LIMIAR_PERCLOS

        if alarme and time.time() - ultimo_bipe >= PAUSA_BIPE:
            ultimo_bipe = time.time()
            tocar_bipe()
            atraso = (time.time() - comeco) * 1000
            print("ALERTA - PERCLOS:", str(round(perclos * 100)) + "%",
                  "| atraso do quadro ate o alarme:", round(atraso, 1), "ms")

        contador_fps += 1
        if time.time() - marca_fps >= 1:
            fps_real = contador_fps / (time.time() - marca_fps)
            contador_fps = 0
            marca_fps = time.time()

        if ear is None:
            texto_ear = "EAR: -"
        else:
            texto_ear = "EAR: " + str(round(ear, 3))

        texto_perclos = "PERCLOS: " + str(round(perclos * 100)) + "%"
        if not janela_cheia:
            texto_perclos += " (enchendo a janela)"

        texto_tempo = ("Proc: " + str(round(np.mean(tempos), 1)) + " ms" +
                       " | FPS: " + str(round(fps_real, 1)))

        linhas = ["Olho: " + texto_olho, texto_ear, texto_perclos, texto_tempo]
        for i, linha in enumerate(linhas):
            cv2.putText(quadro, linha, (10, 30 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        if alarme:
            cv2.putText(quadro, "SONOLENCIA", (10, altura - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

        cv2.imshow("Deteccao de sonolencia", quadro)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            ear_normal = None
            amostras_calib = []
            inicio_calib = time.time()
            janela = []
            print("Calibrando de novo.")

    camera.release()
    cv2.destroyAllWindows()

    if tempos:
        print("Tempo medio por quadro:", round(np.mean(tempos), 1), "ms")


main()