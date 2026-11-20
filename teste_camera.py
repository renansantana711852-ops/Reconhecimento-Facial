import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível acessar a câmera.")
else:
    print("Câmera iniciada com sucesso!")

    while True:
        ret, imagem = camera.read()

        if not ret:
            break

        cv2.imshow("Minha Camera", imagem)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
