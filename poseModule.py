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
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)    
        if self.results.pose_landmarks:
            if draw:
                # Cambia el color de las líneas a verde
                drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, drawing_spec, drawing_spec)
        
        return img
        

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw: 
                    # Dibuja círculos rojos, excepto los 11 y 12 que son azules
                    color = (0, 0, 255) if id in [11, 12] else (0, 0, 255)  # Azul para 11 y 12, rojo para otros
                    radius = 12 if id in [11, 12] else 5
                    cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
                    
                    # Imprime las posiciones de los landmarks 11 y 12
                    if id == 11 or id == 12:
                        print(f'Landmark {id}: ({cx}, {cy})')
        return self.lmList
    

    def findAngle(self, img, p1, p2, p3, draw=True):
        
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # calcular angulo y normalizar para que este en el rango 0, 360
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # ajustar angulo
        if angle < 0:
            angle += 360
        
        # print(f"{angle}°")        
        
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 + 10, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return angle


def main():
    cap = cv2.VideoCapture('./files/dips.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        
        if not success or img is None:
            print("Fin del video o no se pudo leer el frame")
            break
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=True)  # Cambia a True para dibujar los círculos
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
