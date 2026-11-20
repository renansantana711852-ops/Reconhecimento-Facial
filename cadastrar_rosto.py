import cv2
import os

nome = input("Digite o nome da pessoa: ")

pasta = os.path.join("rostos_cadastrados", nome.replace(" ", "_"))
os.makedirs(pasta, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

contador = 1

print("Pressione S para salvar uma foto.")
print("Pressione Q para sair.")

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break

    cv2.imshow("Cadastro de rosto", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('s'):
        caminho = os.path.join(pasta, f"rosto_{contador:02d}.jpg")
        cv2.imwrite(caminho, frame)

        print(f"Foto {contador} salva!")
        contador += 1

    elif tecla == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Cadastro finalizado! Fotos salvas: {contador - 1}")