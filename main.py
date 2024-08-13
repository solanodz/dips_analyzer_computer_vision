import cv2
import numpy as np
import time
import mediapipe as mp
import poseModule as pm
import os
import pandas as pd
import shutil

output_dir = 'resultados_dips'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture("./files/dips.mp4")
detector = pm.poseDetector()
count = 0
dir = 0  # -> 0 = fase concentrica | 1 = fase excentrica

# almacenar los valores mínimos, máximos y velocidades
results = []

# Variables para almacenar la posición y tiempo del punto 13 (hombro izq)
start_position = None
start_time = None
end_position = None
end_time = None
min_angle, max_angle = None, None

# Escala para convertir de píxeles a metros (ajustado a mi situación)
pixel_to_meter_scale = 0.01  #  ajustado para mis 180 cm 

# Configuración del video de salida
output_video_path = os.path.join(output_dir, 'dips_output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break

    # Procesamiento de la pose
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)  # -> brazo izq
        per = np.interp(angle, (200, 280), (0, 100))
        bar = np.interp(angle, (210, 280), (100, 450))

        # mínimos y máximos ángulos
        if min_angle is None or angle < min_angle:
            min_angle = angle
        if max_angle is None or angle > max_angle:
            max_angle = angle

        # Verificar fase del movimiento y calcular velocidad
        if per == 100:
            start_position = lmList[13][1:]  # posición inicial (x, y)
            start_time = time.time()  # tiempo inicial

        if per == 0 and start_position is not None and start_time is not None:
            end_position = lmList[13][1:]  # posición final (x, y)
            end_time = time.time()  # tiempo final

            # distancia en mts
            distance_in_pixels = np.linalg.norm(np.array(end_position) - np.array(start_position))
            distance_in_meters = distance_in_pixels * pixel_to_meter_scale

            time_elapsed = end_time - start_time

            # Calculo de velocidad (m/s)
            speed = distance_in_meters / time_elapsed
            print(f"Velocidad del ciclo {int(count)}: {speed:.2f} m/s")

            # Guardar resultados del ciclo
            results.append({
                'Ciclo': int(count),
                'Minimo': round(min_angle, 2),
                'Maximo': round(max_angle, 2),
                'Velocidad (m/s)': round(speed, 2)
            })

            # reset
            start_position, start_time = None, None
            min_angle, max_angle = None, None  # reinicio para la siguiente rep

        # Check for the dips
        color = (0, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Barra de %
        cv2.rectangle(img, (5, img.shape[0] - 100), (100, img.shape[0] - int(bar)), color, cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (20, img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 1.7, color, 2)

        # Contador
        cv2.rectangle(img, (img.shape[1] - 140, 10), (img.shape[1] - 10, 140), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(int(count)), (img.shape[1] - 105, 105), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 8)
    
    out.write(img)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

output_excel_path = os.path.join(output_dir, 'resultados_dips.xlsx')
df = pd.DataFrame(results)
df.to_excel(output_excel_path, index=False)

# archivo ZIP con la carpeta 'resultados_dips'
shutil.make_archive('resultados_dips', 'zip', output_dir)

print("✅ Video y resultados guardados en:", output_dir)
