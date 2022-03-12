#!/usr/bin/env python3
import cv2
import dlib
import argparse
import numpy as np
from imutils import face_utils
from phase1.ThinPlateSpline import thin_plate_spline_warping
from phase1.Triangulation import triangulation_warping
from phase2.api import PRN
from phase2.prnet import *

def facial_landmark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # detect faces in the grayscale frame
    faceRects = detector(gray, 0)
    face_pts = []
    if len(faceRects) > 0:
        for faceRect in faceRects:
            # compute the bounding box of the face and draw it on the frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(faceRect)
            cv2.rectangle(img, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, faceRect)
            shape = face_utils.shape_to_np(shape)
            # for (i, (x, y)) in enumerate(shape):
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                face_pts.append((x, y))
    else:
        print("ERROR: NO FACES FOUND!!!")

    return len(faceRects), face_pts


def traditional(src, dst, src_face, dst_face, method):
    dst_copy = np.copy(dst)
    hull = cv2.convexHull(np.array(dst_face), returnPoints=False)
    src_hull = []
    dst_hull = []
    for h in hull:
        src_hull.append(src_face[int(h)])
        dst_hull.append(dst_face[int(h)])
    if method == 'tps':
        warped_img = thin_plate_spline_warping(src, dst_copy, src_face, dst_face, dst_hull)
    elif method == 'tri':
        warped_img = triangulation_warping(src, dst, dst_copy, src_hull, dst_hull)
    output = blend(warped_img, dst, dst_hull)

    return output

def blend(warped_img, dst_img, dst_hull):
    hull = []
    for h in dst_hull:
        hull.append((h[0], h[1]))
    
    mask = np.zeros_like(dst_img)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull]))
    center = ((r[0] + int(r[2]/2), r[1] + int(r[3]/2)))

    output = cv2.seamlessClone(np.uint8(warped_img), dst_img, mask, center, cv2.NORMAL_CLONE)

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='prnet', type=str, help='tri, tps, prnet')
    Args = parser.parse_args()
    method = Args.method

    video_path = './TestSet/Test1.mp4'
    face_img_path = './TestSet/Rambo.jpg'
    # video_path = './me.mp4'
    # face_img_path = './ironman.jpg'
    cap = cv2.VideoCapture(video_path)
    face_img = cv2.imread(face_img_path)
    if method != 'prnet':
        # scale_percent = 50 # percent of original size
        # width = int(face_img.shape[1] * scale_percent / 100)
        # height = int(face_img.shape[0] * scale_percent / 100)
        # face_img = cv2.resize(face_img, (width, height), interpolation = cv2.INTER_AREA)
        _, face1_pts = facial_landmark(face_img)
    else:
        prn = PRN(is_dlib = True)
        prev_pos = None
    if (cap.isOpened()== False): 
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            scale_percent = 50 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            if method != "prnet":
                num_face2, face2_pts = facial_landmark(frame)
                if num_face2 == 0:
                    continue
                res = traditional(face_img, frame, face1_pts, face2_pts, method)
            else:
                pos = prn.process(frame)
                ref_pos = prn.process(face_img)

                if pos is None:
                    if prev_pos is not None:
                        pos = prev_pos
                    else:
                        print("ERROR: No Faces Found !!!")
                        res = frame
                if pos is not None:
                    res = deep_learning(prn, pos, ref_pos, frame, face_img)
            cv2.imshow('r', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()