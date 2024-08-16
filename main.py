import cv2
import numpy as np
import time
import mediapipe as mp
import os
import pandas as pd
import shutil
from exercises import exercises
import poseModule as pm

def seleccionar_ejercicio():
    print("Selecciona un ejercicio:")
    for idx, ejercicio in enumerate(exercises.keys(), 1):
        print(f"{idx}. {ejercicio}")

    seleccion = int(input("Introduce el número del ejercicio: ")) - 1
    nombre_ejercicio = list(exercises.keys())[seleccion]
    return exercises[nombre_ejercicio]

def detectar_vista(lmList):
    if not lmList:
        return "Unknown"
    
    left_shoulder_x = lmList[11][1]
    right_shoulder_x = lmList[12][1]
    
    left_hand_x = lmList[15][1]
    right_hand_x = lmList[16][1]
    
    if left_hand_x < right_hand_x and left_shoulder_x < right_shoulder_x:
        return "Izquierda"
    elif left_hand_x > right_hand_x and left_shoulder_x > right_shoulder_x:
        return "Derecha"
    else:
        return "Frontal"

def confirmar_vista(vista_automatica):
    print(f"Vista detectada automáticamente: {vista_automatica}")
    print("Confirma la vista o selecciona una diferente:")
    print("1. Izquierda")
    print("2. Derecha")
    print("3. Frontal")

    seleccion = int(input("Introduce el número de la vista (o presiona Enter para confirmar): "))
    if seleccion == 1:
        return "Izquierda"
    elif seleccion == 2:
        return "Derecha"
    elif seleccion == 3:
        return "Frontal"
    else:
        return vista_automatica

output_dir = 'resultados_exercises'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture("./files/dips.mp4")
detector = pm.poseDetector()
count = 0
dir = 0  # -> 0 = fase concentrica | 1 = fase excentrica

# almacenar los valores mínimos, máximos y velocidades
results = []

# Variables para almacenar la posición y tiempo del punto 13 o 14 (codo izq o der)
start_position = None
start_time = None
end_position = None
end_time = None
min_angle, max_angle = None, None

# Configuración del video de salida
output_video_path = os.path.join(output_dir, 'ejercicio_output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

config = seleccionar_ejercicio()
landmarks_izq = config['landmarks_izq']
landmarks_der = config['landmarks_der']
rango_angulos = config['rango_angulos']
pixel_to_meter_scale = config['escala_pixel_a_metro']

# Leer el primer frame para detectar la vista
success, img = cap.read()
if success:
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    # Detección automática de la vista
    vista_automatica = detectar_vista(lmList)
    vista_seleccionada = confirmar_vista(vista_automatica)

    print(f"Vista seleccionada: {vista_seleccionada}")
    if vista_seleccionada == "Izquierda":
        landmarks = landmarks_izq
        filter_landmarks = set(landmarks_izq)
        color_landmarks = (255, 0, 0)  # Azul para los puntos relevantes del lado izquierdo
        selected_side = 'Izquierdo'
    elif vista_seleccionada == "Derecha":
        landmarks = landmarks_der
        filter_landmarks = set(landmarks_der)
        color_landmarks = (0, 255, 0)  # Verde para los puntos relevantes del lado derecho
        selected_side = 'Derecho'
    else:
        # Manejar la vista frontal si es necesario, o usar una combinación de ambos
        landmarks = landmarks_izq  # O combinar los landmarks según el ejercicio
        filter_landmarks = set(landmarks_izq + landmarks_der)
        color_landmarks = (0, 255, 255)  # Amarillo para los puntos relevantes en vista frontal
        selected_side = None  # En vista frontal, mostramos ambos lados si es necesario

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar el video desde el principio

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        for lado, landmarks in [('Izquierdo', landmarks_izq), ('Derecho', landmarks_der)]:
            angle = detector.findAngle(img, *landmarks)
            per = np.interp(angle, rango_angulos, (0, 100))
            bar = np.interp(angle, (rango_angulos[0] + 10, rango_angulos[1]), (100, 450))

            # mínimos y máximos ángulos
            if min_angle is None or angle < min_angle:
                min_angle = angle
            if max_angle is None or angle > max_angle:
                max_angle = angle

            # Verificar fase del movimiento y calcular velocidad
            if per == 100:
                start_position = lmList[landmarks[1]][1:]  # posición inicial (x, y)
                start_time = time.time()  # tiempo inicial

            if per == 0 and start_position is not None and start_time is not None:
                end_position = lmList[landmarks[1]][1:]  # posición final (x, y)
                end_time = time.time()  # tiempo final

                # Calculo de la velocidad
                distance_in_pixels = np.linalg.norm(np.array(end_position) - np.array(start_position))
                distance_in_meters = distance_in_pixels * pixel_to_meter_scale
                time_elapsed = end_time - start_time
                speed = distance_in_meters / time_elapsed

                # Guardar resultados del ciclo
                results.append({
                    'Ciclo': int(count),
                    'Lado': lado,
                    'Minimo': round(min_angle, 2),
                    'Maximo': round(max_angle, 2),
                    'Velocidad (m/s)': round(speed, 2)
                })

                # reset
                start_position, start_time = None, None
                min_angle, max_angle = None, None

            if selected_side == lado or selected_side is None:
                # Mostrar solo para el lado seleccionado o ambos en vista frontal
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

                cv2.rectangle(img, (5, img.shape[0] - 100), (100, img.shape[0] - int(bar)), color, cv2.FILLED)
                cv2.putText(img, f"{int(per)}% {lado}", (20, img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 1.7, color, 2)

                # Contador para el lado seleccionado
                cv2.rectangle(img, (img.shape[1] - 140, 10), (img.shape[1] - 10, 140), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, str(int(count)), (img.shape[1] - 105, 105), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 8)

    # Filtrar los landmarks visibles según la vista seleccionada
    if filter_landmarks:
        for id, x, y in lmList:
            if id in filter_landmarks:
                cv2.circle(img, (x, y), 5, color_landmarks, cv2.FILLED)

    out.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

output_excel_path = os.path.join(output_dir, f'resultados_{config["nombre"]}.xlsx')
df = pd.DataFrame(results)
df.to_excel(output_excel_path, index=False)

# archivo ZIP con la carpeta 'resultados_exercises'
shutil.make_archive(output_dir, 'zip', output_dir)

print("✅ Video y resultados guardados en:", output_dir)
