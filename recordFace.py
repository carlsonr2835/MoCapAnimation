import cv2
from matplotlib import pyplot

def liveCapture():
    #set camera to the default system camera, the webcam, for my machine at least
    webcam = cv2.VideoCapture(0)

    #until program is quit, continuously read and show the webcam footage
    while webcam.isOpened():
        success, imgage = webcam.read()
        if not success:
            print("Big poop. Data cannot be read from the webcam :(")
            return

        cv2.imshow("Ryan", imgage)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break
            
    webcam.release()
    cv2.destroyAllWindows()