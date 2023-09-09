import numpy as np
import cv2

import urllib.request
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def countF(imglink):
    req = urllib.request.urlopen(imglink)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    # img = cv2.imread(response.content)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=8,     
        minSize=(20, 20)
    )
    return len(faces)

# print(countF('https://media.cnn.com/api/v1/images/stellar/prod/230616115528-barack-obama-2022-file.jpg?c=original'))

# cap = cv2.VideoCapture(0)

# while True:
#     ret, img = cap.read()
#     if not ret:
#         break
#     img = cv2.flip(img,1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(
#         gray,     
#         scaleFactor=1.2,
#         minNeighbors=8,     
#         minSize=(20, 20)
#     )
#     count = len(faces)
#     cv2.putText(img, f"{count} number of persons", (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2,cv2.LINE_AA)
#     print(f'{count} number of persons')
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]  
#         # cv2.imshow('test', roi_color)
#     cv2.imshow('video',img)
#     k = cv2.waitKey(1)
#     if k == ord('q'): 
#         break
# cap.release()
# cv2.destroyAllWindows()