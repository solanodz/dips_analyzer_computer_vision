import cv2
import time 
import mediapipe as mp
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
        # Variables para contar repeticiones y almacenar ángulos
        self.direction = 0  # 0: no movimiento, 1: bajando, 2: subiendo
        self.rep_count = 0
        self.min_angle = 180
        self.max_angle = 0
        self.reps = []  # Lista para almacenar (mínimo, máximo) de cada repetición
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)    
        if self.results.pose_landmarks:
            if draw:
                drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, drawing_spec, drawing_spec)
        
        return img
        

    def findPosition(self, img, exercise_landmarks=None, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                if draw: 
                    # Cambiar el color según los landmarks del ejercicio
                    if exercise_landmarks and id in exercise_landmarks:
                        color = (255, 0, 0)  # Azul para los puntos relevantes del ejercicio
                    else:
                        color = (0, 0, 255)  # Rojo para los otros puntos

                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return self.lmList
    

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 + 10, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return angle

    def trackRepetitions(self, angle, threshold_min, threshold_max):
        # Actualiza el ángulo mínimo y máximo
        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)
        
        # Detecta el cambio de dirección para contar una repetición
        if self.direction == 0 and angle > threshold_max:
            self.direction = 1  # Inicia bajada
        elif self.direction == 1 and angle < threshold_min:
            self.direction = 2  # Inicia subida
        elif self.direction == 2 and angle > threshold_max:
            self.direction = 0  # Repetición completada
            self.rep_count += 1
            self.reps.append((self.min_angle, self.max_angle))
            self.min_angle = 180  # Reiniciar el mínimo
            self.max_angle = 0  # Reiniciar el máximo
                
        return self.rep_count, self.reps

# Ejemplo de uso:

def main():
    cap = cv2.VideoCapture('./files/dips.mp4')
    detector = poseDetector()
    
    exercise_landmarks = [11, 13, 15]  # Landmarks relevantes para dips en el lado izquierdo
    threshold_min = 60
    threshold_max = 160

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, exercise_landmarks=exercise_landmarks, draw=True)
        
        if len(lmList) > 0:
            # Calcula el ángulo clave para este ejercicio
            angle = detector.findAngle(img, 11, 13, 15)  # Por ejemplo, el ángulo del codo izquierdo en un dip
            
            # Cuenta repeticiones y guarda ángulos mínimos y máximos
            rep_count, reps = detector.trackRepetitions(angle, threshold_min, threshold_max)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"Repeticiones: {rep_count}")
    print("Ángulos por repetición (mín, máx):", reps)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()