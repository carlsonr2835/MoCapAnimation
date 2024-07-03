"""
Webcam access code from nicholas renotte

face mesh drawing code from Koolac https://www.youtube.com/watch?v=yvXPKfil1hY
    -> I'm having such a weird bug with this inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
"""

import cv2
import mediapipe as mp
faceMesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
drawingStyles = mp.solutions.drawing_styles

def liveCapture():
    #set camera to the default system camera, the webcam, for my machine at least
    webcam = cv2.VideoCapture(0)

    #until program is quit, continuously read and show the webcam footage
    while webcam.isOpened():
        success, capture = webcam.read()
        if not success:
            print("Big poop. Data cannot be read from the webcam :(")
            return

        #openCV used BGR and mediapipe uses RGB. Convert the image's colors to RGB and find the facemesh
        capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
        #this line makes the error happen
        meshResult = faceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True).process(capture)

        #annotate the capture with the facemesh IF a face has been detected
        #convert the colors back to opencv BGR
        capture = cv2.cvtColor(capture, cv2.COLOR_RGB2BGR)
        if meshResult.multi_face_landmarks:
            #draw annotation on all faces detected
            for face_landmarks in meshResult.multi_face_landmarks:
                #draw iris landmarks
                drawing.draw_landmarks(
                    image=capture,
                    landmark_list = face_landmarks,
                    connections = faceMesh.FACEMESH_IRISES,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawingStyles.get_default_face_mesh_iris_connections_style()
                )

                #draw contour landmarks
                drawing.draw_landmarks(
                    image=capture,
                    landmark_list = face_landmarks,
                    connections = faceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawingStyles.get_default_face_mesh_contours_style()
                )

                #draw tesselation
                drawing.draw_landmarks(
                    image=capture,
                    landmark_list = face_landmarks,
                    connections = faceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawingStyles.get_default_face_mesh_tesselation_style()
                )


        #draw the webcam capture
        cv2.imshow("Capture", capture)
        #quit window upon button press q
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    #when done, webcam window goes away        
    webcam.release()
    cv2.destroyAllWindows()