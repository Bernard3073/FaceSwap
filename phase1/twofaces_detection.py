import cv2
import dlib
from imutils import face_utils

def twofaces_detection(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    rects = detector(gray,1)

    points = []
    faces = []

    num_faces = len(rects)

    if(num_faces==2):
        for (_,rect) in enumerate(rects):
            
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            (x,y,_,_) = face_utils.rect_to_bb(rect)

            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            for (x,y) in shape:
                # cv2.circle(img,(x,y),2,(0,0,255),-1)
                points.append((x,y))

            faces.append(points)
            points = []
            
    return num_faces,faces