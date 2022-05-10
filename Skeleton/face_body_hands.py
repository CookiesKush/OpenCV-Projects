import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        # Face
        mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_CONTOURS)
        
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(7, 118, 5), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
        # Left hand                        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(7, 118, 5), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        # Body
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0),thickness=(2),circle_radius=(2)),
                                  mp_drawing.DrawingSpec(color=(255,0,0),thickness=(2),circle_radius=(2)))
        
        
        cv2.imshow('Hand & Body & Face Tracking (press Q to exit)', image)
        
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()