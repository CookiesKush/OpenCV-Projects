import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(255, 0, 89), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(4, 159, 14), thickness=2, circle_radius=2),
                                         )
            
        for landmark in mp_hands.HandLandmark:
            if landmark.value == "MIDDLE_FINGER_TIP 12":
                mp_drawing.DrawingSpec(color=(4, 159, 14), thickness=2, circle_radius=2)
        
        cv2.imshow('Green Hand Tracking (press Q to exit)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()