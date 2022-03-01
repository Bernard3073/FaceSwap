import cv2
import dlib
import argparse

def main():
    video_path = './TestSet/Test1.mp4'
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('r', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()