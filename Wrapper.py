import cv2
import dlib
import argparse
import numpy as np
import scipy

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

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def delauney_triangle(img, faces):
    box = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(box) # cv2.Subdiv2D(tuple(box)) 
    # insert points into subdiv
    for f in faces:
        subdiv.insert(tuple(f))
    triangle_list = subdiv.getTriangleList()

    face_pts = dict()
    for i, j in enumerate(faces):
        k = tuple(j)
        face_pts[k] = i

    for t in range(len(triangle_list)):
        pt_1 = (t[0], t[1])
        pt_2 = (t[2], t[3])
        pt_3 = (t[4], t[5])
        pts = [pt_1, pt_2, pt_3]

        if rect_contains(faces, pt_1) and rect_contains(faces, pt_2) and rect_contains(faces, pt_3):
            indices = []
            res = []
            for p in pts:
                idx = face_pts.get(tuple(p), False)
                if idx:
                    indices.append(idx)
            # three points form a delauney triangle        
            if len(indices) == 3:
                res.append(np.array(indices))
    return res

def triangulation_warping(src, src_tri, dst_tri):
    # compute Barycentric coordinates for each triangle in the destination
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    x, y, w, h = dst_rect
    tri_coord_x, tri_coord_y = np.mgrid[x:x+w, y:y+h].reshape(2,-1)
    B_mat = [dst_tri[0][0], dst_tri[1][0], dst_tri[2][0],
            dst_tri[0][1], dst_tri[1][1], dst_tri[2][1]]
    B_mat = np.array(B_mat).reshape(2, 3)
    B_mat = np.vstack(B_mat, [1, 1, 1])
    B_mat_inv = np.linalg.inv(B_mat)
    ones = np.ones(1, tri_coord_y.shape[0])
    tri_coord_homo = np.vstack((tri_coord_x, tri_coord_y, ones))
    barycen_cord = B_mat_inv @ tri_coord_homo
    alpha, beta, gamma = barycen_cord[0], barycen_cord[1], barycen_cord[2]
    # check if a point lies inside the triangle
    inliers = []
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and 0 <= alpha + beta + gamma <= 1:
        inliers.append([alpha, beta, gamma])
    A_mat = [src_tri[0][0], src_tri[1][0], src_tri[2][0],
            src_tri[0][1], src_tri[1][1], src_tri[2][1]]
    A_mat = np.array(A_mat).reshape(2, 3)
    A_mat = np.vstack(A_mat, [1, 1, 1])
    [src_x, src_y, src_z] = A_mat @ np.array(inliers)
    src_x = src_x / src_z
    src_y = src_y / src_z
    # copy back the value of pixel at (x_A, y_A) to the target location
    dst = scipy.interpolate.interp2d(src_x, src_y, src_z, kind='cubic')
    return dst

def U(r):
    res = (r**2) * np.log(r**2)
    return res

def tps_model(src_1d, dst_pts):
    p = len(dst_pts)
    K_mat = np.zeros((p, p), np.float32) # size: p x p
    P_mat = np.zeros((p, 3), np.float32) # size: p x 3
    for i in range(p):
        for j in range(p):
            K_mat[i, j] = U(np.linalg.norm(dst_pts[i, i] - dst_pts[j, j]))
    P_mat = np.hstack((P_mat, np.ones((dst_pts, 1))))
    
    K_P_Pt_mat_row1 = np.hstack((K_mat, P_mat))
    K_P_Pt_mat_row2 = np.hstack((P_mat.T, np.zeros((P_mat.T.shape[0], P_mat.T.shape[1]))))
    K_P_Pt_mat = np.vstack((K_P_Pt_mat_row1, K_P_Pt_mat_row2))
    lamda = 0.0001
    A_mat = K_P_Pt_mat + lamda * np.identity((p + 3))
    v_o = np.concatenate((src_1d, [0, 0, 0]))
    est_params = np.linalg.inv(A_mat) @ v_o
    return est_params

def f(x_param, y_param, p, dst_pts, img_warp_row, img_warp_col):
    ax_x, ay_x, a1_x = x_param[p], x_param[p+1], x_param[p+2]
    ax_y, ay_y, a1_y = y_param[p], y_param[p+1], y_param[p+2]
    f1, f2 = 0, 0
    for i in range(p):
        f1 += x_param[i] * U(np.linalg.norm(dst_pts[i] - img_warp_row))
        f2 += y_param[i] * U(np.linalg.norm(dst_pts[i] - img_warp_col))
    f1 += a1_x + ax_x * img_warp_row + ay_x * img_warp_col
    f2 += a1_y + ax_y * img_warp_row + ay_y * img_warp_col
    return f1, f2

def thin_plate_spline_warping(src, dst, src_pts, dst_pts):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    p = len(src_pts)
    r = cv2.boundingRect(np.float32([dst_pts]))
    mask = np.zeros((r[3], r[2], 3), np.float32)
    warped_img = np.copy(mask)

    est_params_x = tps_model(src[:, 0], dst_pts)
    est_params_y = tps_model(src[:, 1], dst_pts)
    for i in range(warped_img.shape[1]):
        for j in range(warped_img.shape[0]):
            f1, f2 = f(est_params_x, est_params_y, p, dst_pts, i + r[0], j + r[1])
            warped_img[j, i] = src[f2, f1, :] 
    warped_img *= mask
    dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( (1.0, 1.0, 1.0) - mask )
    dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] + warped_img
    return dst

def traditional(src, dst, src_pts, dst_pts, method):
    if method == 'tri':
        res = triangulation_warping(src, src_pts, dst_pts)
    elif method == 'tps':
        res = thin_plate_spline_warping(src, dst, src_pts, dst_pts)
    return res

def main():
    video_path = './TestSet/Test2.mp4'
    face_img_path = './TestSet/Rambo.jpg'
    cap = cv2.VideoCapture(video_path)
    face_img = cv2.imread(face_img_path)
    face1, face1_pts = facial_landmark(face_img)

    if (cap.isOpened()== False): 
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            face2, face2_pts = facial_landmark(frame)
            res = traditional(face1, face1_pts, face2, face2_pts)
            cv2.imshow('r', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()