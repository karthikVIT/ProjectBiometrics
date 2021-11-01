import cv2
import numpy

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

user_name = input("Enter the Name: ")
count = 0

while(True):
    ret, color_img = camera.read()
    gray_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite("DataSet/User."+str(user_name)+"." +
                    str(count)+".jpg", gray_image[y:y+h, x:x+w])
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.waitKey(100)

    cv2.imshow("Face Detection", color_img)
    cv2.waitKey(1)

    if(count > 15):
        break

camera.release()
cv2.destroyAllWindows()
