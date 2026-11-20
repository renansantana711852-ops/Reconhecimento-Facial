import cv2

camera = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, imagem = camera.read()

    if not ret:
        break

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    rostos = detector.detectMultiScale(
        imagem_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, largura, altura) in rostos:
        cv2.rectangle(
            imagem,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )

    cv2.imshow("Deteccao Facial", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()