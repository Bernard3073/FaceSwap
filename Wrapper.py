#!/usr/bin/env python
import cv2
from matplotlib import pyplot as plt
import dlib
import argparse
import numpy as np
from imutils import face_utils, rotate
from phase1.ThinPlateSpline import thin_plate_spline_warping
from phase1.Triangulation import triangulation_warping
# from phase2.api import PRN
# from phase2.prnet import *
from phase1.twofaces_detection import twofaces_detection
# from phase2.prnet_twofaces import *

def facial_landmark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # detect faces in the grayscale frame
    faceRects = detector(gray, 1)
    
    face_pts = []
    if len(faceRects) > 0:
        for faceRect in faceRects:
            # compute the bounding box of the face and draw it on the frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(faceRect)
            # cv2.rectangle(img, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, faceRect)
            shape = face_utils.shape_to_np(shape)
            # for (i, (x, y)) in enumerate(shape):
            for (x, y) in shape:
                # cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                face_pts.append((x, y))


    return len(faceRects), face_pts


def traditional(src, dst, src_face, dst_face, method):
    dst_copy = np.copy(dst)
    hull = cv2.convexHull(np.array(dst_face), returnPoints=False)
    src_hull = []
    dst_hull = []
    for h in hull:
        src_hull.append(src_face[int(h)])
        dst_hull.append(dst_face[int(h)])
    if method == 'TPS':
        warped_img = thin_plate_spline_warping(src, dst_copy, src_face, dst_face, dst_hull)
    elif method == 'Tri':
        warped_img = triangulation_warping(src, dst, dst_copy, src_hull, dst_hull)
    output = blend(warped_img, dst, dst_hull)

    return output

def blend(warped_img, dst_img, dst_hull):
    hull = []
    for h in dst_hull:
        hull.append((h[0], h[1]))
    
    mask = np.zeros_like(dst_img)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
    # shrink mask to fit the face
    kernel = np.ones((6, 6))
    mask = cv2.erode(mask, kernel)
    (x, y, w, h) = cv2.boundingRect(np.float32([hull]))
    dst_face_center = (int((x + x + w)/2), int((y + y + h)/2))

    output = cv2.seamlessClone(np.uint8(warped_img), dst_img, mask, dst_face_center, cv2.NORMAL_CLONE)

    # motion filtering using bilateral filter
    face = output[y:y+h, x:x+w]
    output[y:y+h, x:x+w] = cv2.bilateralFilter(face, 9, 75, 75)

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='Tri', type=str, help='Tri, TPS, PRNet')
    parser.add_argument('--videopath', default='./Data/Data3.mp4', help='path of the video')
    parser.add_argument('--facepath', default='./TestSet/Rambo.jpg', help='path of the celebrity face to swap')
    parser.add_argument('--twofaces', default=True, type=bool, help='swap two faces in video')
    parser.add_argument('--debug', default=False, type=bool, help='only process first frame for testing')
    Args = parser.parse_args()
    method = Args.method
    twofaces = Args.twofaces
    video_path = Args.videopath
    face_img_path = Args.facepath
    debug = Args.debug

    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    if (cap.isOpened()== False): 
        print("Error opening video file")
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(video_path[-9:-4]+'Output'+method+'.mp4',  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            20, (frame_width, frame_height)) 
    face_img = cv2.imread(face_img_path)
    count = 0
    if(not twofaces):
        if method != 'PRNet':
            _, face1_pts = facial_landmark(face_img)
        else:
            prn = PRN(is_dlib = True)
            prev_pos = None

        while(True):
            ret, frame = cap.read()  
            if not ret:
                break
            if ret == True:
                if method != "PRNet":
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
                            print("Frame " + count + ": No Faces Found !!!")
                            res = frame
                    if pos is not None:
                        res = deep_learning(prn, pos, ref_pos, frame, face_img)
                cv2.imshow('r', res)
                result.write(res)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    else:
        print("Swap two faces in video !!!")
        if(method == 'PRNet'):
            prn = PRN_twofaces(is_dlib=True)
            prev_pos = None
        while(True):
            ret, frame = cap.read()
            if not ret: 
                break
            if(ret == True):
                # frame = rotate(frame, 180)
                if(method != 'PRNet'):
                    num_faces, twoface_pts = twofaces_detection(frame)
                    if(num_faces < 2):
                        print("Frame " + count + ": Not able to find two faces in video !!!")
                        continue
                    else:
                        # Take greatest two faces
                        face1_pts = twoface_pts[0]
                        face2_pts = twoface_pts[1]
                    tmp = traditional(frame, frame, face1_pts, face2_pts, method)
                    res = traditional(frame, tmp, face2_pts, face1_pts, method)

                    cv2.imshow('r', res)
                    result.write(res)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    pos = prn.process(frame)
                    if pos is None:
                        pos = prev_pos
                    if len(pos) == 2:
                        prev_pos = pos
                        pose1 ,pose2 = pos[0],pos[1]
                        tmp = deep_learning(prn, pose1, pose2, frame, frame)
                        res = deep_learning(prn, pose2, pose1, tmp, frame)
                    else:
                        pos = prev_pos
                    cv2.imshow('r', res)
                    result.write(res)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
            if debug == True:
                break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()