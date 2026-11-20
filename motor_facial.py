"""Motor de reconhecimento facial (detecção + embedding + comparação).

Usa dois modelos leves que vêm junto com o OpenCV (pasta `modelos/`):
  - YuNet : detecta rostos (bem melhor que Haar para rostos pequenos/distantes
            e com leve inclinação) e devolve 5 pontos de referência.
  - SFace : transforma o rosto alinhado em um vetor de 128 números.
Dois rostos da mesma pessoa geram vetores parecidos (similaridade cosseno alta).
"""
import os
import threading
from pathlib import Path

import cv2
import numpy as np

PASTA_MODELOS = Path(__file__).parent / "modelos"
ARQ_DETECTOR = PASTA_MODELOS / "face_detection_yunet_2023mar.onnx"
ARQ_RECONHECEDOR = PASTA_MODELOS / "face_recognition_sface_2021dec.onnx"


def _env(nome, padrao):
    return type(padrao)(os.environ.get(nome, padrao))


# ----------------------------------------------------------------- ajustes ---
# Tudo pode ser ajustado por variável de ambiente sem mexer no código.
LIMIAR_SIMILARIDADE = _env("LIMIAR", 0.40)   # abaixo disso = desconhecido
                                              # (SFace recomenda 0.363; usamos um pouco
                                              #  mais rígido porque é controle de acesso)
MARGEM_SEGUNDO = _env("MARGEM", 0.05)        # o 1º colocado precisa ganhar do 2º por isso
LARGURA_MIN_RECONHECER = _env("LARGURA_MIN_RECONHECER", 60)  # px do rosto no frame
LARGURA_MIN_CADASTRO = _env("LARGURA_MIN_CADASTRO", 130)
LARGURA_IDEAL = _env("LARGURA_IDEAL", 180)
NITIDEZ_MIN = _env("NITIDEZ_MIN", 25.0)
BRILHO_MIN = _env("BRILHO_MIN", 55)
BRILHO_MAX = _env("BRILHO_MAX", 215)
SCORE_DETECCAO = _env("SCORE_DETECCAO", 0.6)


class MotorFacial:
    def __init__(self):
        for arq in (ARQ_DETECTOR, ARQ_RECONHECEDOR):
            if not arq.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado: {arq}. Rode `python baixar_modelos.py`."
                )
        self._trava = threading.Lock()  # os modelos do OpenCV não são thread-safe
        self._detector = cv2.FaceDetectorYN.create(
            str(ARQ_DETECTOR), "", (640, 480), SCORE_DETECCAO, 0.3, 10
        )
        self._reconhecedor = cv2.FaceRecognizerSF.create(str(ARQ_RECONHECEDOR), "")

    # ---------------------------------------------------------- detecção ---
    def detectar(self, img):
        """Retorna as linhas do YuNet (15 valores) ordenadas do maior rosto pro menor."""
        altura, largura = img.shape[:2]
        with self._trava:
            self._detector.setInputSize((largura, altura))
            _, rostos = self._detector.detect(img)
        if rostos is None:
            return []
        return sorted(rostos, key=lambda r: r[2] * r[3], reverse=True)

    # --------------------------------------------------------- embedding ---
    def embedding(self, img, rosto):
        with self._trava:
            alinhado = self._reconhecedor.alignCrop(img, rosto)
            vetor = self._reconhecedor.feature(alinhado)
        vetor = vetor.flatten().astype(np.float32)
        return vetor / (np.linalg.norm(vetor) + 1e-9)

    # ----------------------------------------------------------- análise ---
    def analisar(self, img):
        """Detecta o maior rosto e mede a qualidade da imagem dele.

        Devolve um dict com: rosto (linha crua), caixa normalizada (0-1), largura em px,
        pose (-1..1, positivo = virou para o lado esquerdo da pessoa), nitidez, brilho,
        n_rostos e `problemas` (lista de códigos: pequeno, desfocado, escuro, claro).
        """
        altura, largura_img = img.shape[:2]
        rostos = self.detectar(img)
        info = {"n_rostos": len(rostos), "rosto": None, "problemas": []}
        if not rostos:
            return info

        r = rostos[0]
        x, y, w, h = [float(v) for v in r[:4]]
        olho_d, olho_e, nariz = r[4:6], r[6:8], r[8:10]
        dist_olhos = max(abs(olho_e[0] - olho_d[0]), 1.0)
        meio_olhos_x = (olho_d[0] + olho_e[0]) / 2
        pose = float(np.clip((nariz[0] - meio_olhos_x) / dist_olhos, -1, 1))

        x0, y0 = max(int(x), 0), max(int(y), 0)
        recorte = img[y0:int(y + h), x0:int(x + w)]
        nitidez = brilho = 0.0
        if recorte.size:
            cinza = cv2.cvtColor(cv2.resize(recorte, (112, 112)), cv2.COLOR_BGR2GRAY)
            nitidez = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
            brilho = float(cinza.mean())

        info.update(
            rosto=r,
            caixa=[x / largura_img, y / altura, w / largura_img, h / altura],
            largura=w,
            pose=pose,
            nitidez=nitidez,
            brilho=brilho,
            score_deteccao=float(r[14]),
        )
        return info

    @staticmethod
    def problemas_reconhecimento(info, largura_min=LARGURA_MIN_RECONHECER):
        p = []
        if info["largura"] < largura_min:
            p.append("pequeno")
        if info["nitidez"] < NITIDEZ_MIN:
            p.append("desfocado")
        if info["brilho"] < BRILHO_MIN:
            p.append("escuro")
        elif info["brilho"] > BRILHO_MAX:
            p.append("claro")
        return p


class BaseDeRostos:
    """Mantém os embeddings em memória e compara um rosto novo com todos."""

    def __init__(self):
        self._ids = np.zeros(0, dtype=np.int64)
        self._matriz = np.zeros((0, 128), dtype=np.float32)

    def carregar(self, pares):
        """pares: lista de (pessoa_id, vetor)"""
        if pares:
            self._ids = np.array([p for p, _ in pares], dtype=np.int64)
            self._matriz = np.stack([v for _, v in pares]).astype(np.float32)
        else:
            self._ids = np.zeros(0, dtype=np.int64)
            self._matriz = np.zeros((0, 128), dtype=np.float32)

    def comparar(self, vetor):
        """Retorna (pessoa_id | None, score, segundo_score).

        O score de cada pessoa é a média das 3 melhores amostras dela: um único frame
        parecido por acaso não basta, e uma amostra ruim não derruba quem tem boas.
        """
        if len(self._ids) == 0:
            return None, 0.0, 0.0
        sims = self._matriz @ vetor
        por_pessoa = {}
        for pid in np.unique(self._ids):
            s = np.sort(sims[self._ids == pid])[::-1][:3]
            por_pessoa[int(pid)] = float(s.mean())
        ordem = sorted(por_pessoa.items(), key=lambda kv: kv[1], reverse=True)
        melhor_id, melhor = ordem[0]
        segundo = ordem[1][1] if len(ordem) > 1 else -1.0
        return melhor_id, melhor, segundo

    @staticmethod
    def aceito(score, segundo):
        return score >= LIMIAR_SIMILARIDADE and (score - segundo) >= MARGEM_SEGUNDO
