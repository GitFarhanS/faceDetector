import cv2

cap = cv2.VideoCapture(0)
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    #turns image to grayscale
    grayScaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face, detected objects are returned as a list of rectangles   
    faceCoordinates = trainedFaceData.detectMultiScale(grayScaleImg)
    try: 
        for (x,y,w,h) in faceCoordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    except IndexError:
        # Handle the case where no faces are detected
        pass
    
    cv2.imshow('Face detector', frame)
    print("complete")

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()