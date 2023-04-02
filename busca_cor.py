import cv2
import numpy as np
import time

captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

verme_inf = (170, 120, 70)
verme_sup = (180, 255, 255)

# Variáveis para medir o tempo
quadros = 0
tempo_inicial = time.time()

while True:
    # A variável ret é um valor booleano que será True se conseguir capturar o vídeo
    ret, frame = captura.read()
    if ret:
        # Inverter o sentido da camera verticalmente
        frame = cv2.flip(frame, 0)
        #cv2.imshow('Video', frame)
        # Converter a imagem para o espaço de cores HSV
        hsv_imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Aplicar um limiar para segmentar a imagem, mantendo apenas os pixels na faixa de valores de cor rosa
        mask = cv2.inRange(hsv_imagem, verme_inf, verme_sup)

        # Aplicar operação de abertura para remover ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_open = cv2.dilate(mask_open, kernel_dilate, iterations=8)
        
        contornos, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

        # Calculando o momento e o centroide
        for contorno in contornos:
            if cv2.contourArea(contorno) < 1800:
                pass
            else:
                # Desenhar o contorno da imagem
                # cv2.drawContours(frame, contorno, -1, (255,0,0), 3)
                # Desenhar retângulo de bounding box
                x,y,w,h = cv2.boundingRect(contorno)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                
                # print(cv2.contourArea(contorno))
                M = cv2.moments(contorno)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # Desenhando círculo no centro da figura
                cv2.circle(frame, (cx,cy), 10, (0,255,255), -1)
                texto = f'X: {cx}     Y: {cy}'
                if cx < 200:
                    print("Esquerda")
                elif cx > 440:
                    print("Direita")
                else:
                    pass
                #print(texto)
                # Exibindo coordenadas do centro da figura no canto ba bounding box
                #cv2.putText(frame, texto, (x+10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)

        # cv2.imshow("Alvo", frame)
        # cv2.imshow("Segmentação", mask_open)
        mask_open = cv2.cvtColor(mask_open, cv2.COLOR_GRAY2BGR)

        uniao = np.hstack((frame, mask_open))
        cv2.imshow("Alvo", uniao)
        quadros += 1

        # Verificar o tempo desde o último segundo
        tempo_atual = time.time()
        tempo_passado = tempo_atual - tempo_inicial

        # Se passou um segundo
        if tempo_passado >= 1.0:
            fps = quadros / tempo_passado
            print('FPS:', round(fps, 2))
            quadros = 0
            tempo_inicial = tempo_atual
    if cv2.waitKey(1) == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
