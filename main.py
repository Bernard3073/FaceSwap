import cv2
import dlib
import argparse

def facial_landmark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(img, 0)
    bboxes = []
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        cvRect = [int(x1), int(y1), int(x2), int(y2)]
        bboxes.append(cvRect)
        cv2.rectangle(img, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3])
        , (0, 255, 0), int(round(img.shape[0]/150)), 4)
    return img, bboxes
    
def main():
    video_path = './TestSet/Test2.mp4'
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            res, bboxes = facial_landmark(frame)
            cv2.imshow('r', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()