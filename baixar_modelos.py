"""Baixa os modelos ONNX (YuNet + SFace) caso a pasta `modelos/` esteja vazia."""
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
ARQUIVOS = {
    "face_detection_yunet_2023mar.onnx": BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
pasta = Path(__file__).parent / "modelos"
pasta.mkdir(exist_ok=True)
for nome, url in ARQUIVOS.items():
    destino = pasta / nome
    if destino.exists() and destino.stat().st_size > 100_000:
        print("ok  ", nome)
        continue
    print("baixando", nome)
    urllib.request.urlretrieve(url, destino)
