import cv2 as cv       
import numpy as np

face_classifier = cv.CascadeClassifier('D:/program/vscode/Project/Face_recognition/har_face.xml')

#to extract property of an image
# img is capture in rgb but we will convert into Gray scale

def face_extractor(img):

    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY) 
    face = face_classifier.detectMultiScale(gray , 1.3,5)

    if face is ( ):
        return None
    #Cropping of face
    for (x,y,w,h) in face:
        crop_face = img[y:y+h , x:x+w]

    return  crop_face


cap = cv.VideoCapture(0)
count = 0 

while True:
    ret , frame = cap.read()

    if face_extractor(frame) is not None:
        count = count + 1
        face = cv.resize(face_extractor(frame), (200,200))
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

        file_name_path = 'D:/program/vscode/Project/Face_recognition/Photos/user.' +'1.'+ str(count) + '.jpg'
        cv.imwrite(file_name_path,face)

        cv.putText(face , str(count),(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv.imshow('face cropper',face)


    else:
        print("face not found (^.^)")
        pass
    if cv.waitKey(1)==13 or count == 100:
        break
cap.release()
cv.destroyAllWindows()
print("data set collection complete")
