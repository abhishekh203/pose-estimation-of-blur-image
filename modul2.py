import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

image = cv2.imread('as.jpeg')


resized_image = cv2.resize(image, (0, 0), fx=3, fy=3)


resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)


results = pose.process(resized_image_rgb)


if results.pose_landmarks:
    mp_drawing = mp.solutions.drawing_utils
    annotated_image = resized_image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cv2.imshow('Pose Detection - Resized Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


pose.close()
