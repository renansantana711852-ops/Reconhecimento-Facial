import cv2
import os
from deepface import DeepFace

pasta_rostos = "rostos_cadastrados"

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

contador_quadros = 0
nome_reconhecido = "Analisando..."

print("Reconhecimento facial iniciado!")
print("Pressione Q para sair.")

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostos = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in rostos:

        rosto = frame[y:y+h, x:x+w]

        if contador_quadros % 10 == 0:

            try:
                resultados = DeepFace.find(
                    img_path=rosto,
                    db_path=pasta_rostos,
                    model_name="Facenet",
                    detector_backend="skip",
                    distance_metric="cosine",
                    enforce_detection=False,
                    silent=True
                )

                resultado = resultados[0]

                if not resultado.empty:
                    caminho = resultado.iloc[0]["identity"]
                    nome_reconhecido = os.path.basename(
                        os.path.dirname(caminho)
                    )
                else:
                    nome_reconhecido = "Desconhecido"

            except Exception as erro:
                print("Erro no reconhecimento:", erro)
                nome_reconhecido = "Desconhecido"

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            nome_reconhecido,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    contador_quadros += 1

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()