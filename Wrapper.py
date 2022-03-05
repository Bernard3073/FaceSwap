import cv2
import dlib
import argparse
import numpy as np
import scipy
from imutils import face_utils

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

    return (bX, bY, bW, bH), face_pts

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

def triangulation_warping(src, dst, src_tri, dst_tri, dst_hull):
    # compute Barycentric coordinates for each triangle in the destination
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
    if np.isnan(res):
        res = 0
    return res

def tps_model(src_1d, dst_pts):
    p = len(dst_pts)
    K_mat = np.zeros((p, p), np.float32) # size: p x p
    P_mat = np.zeros((p, 3), np.float32) # size: p x 3
    for i in range(p):
        for j in range(p):
            K_mat[i, j] = U(np.linalg.norm(dst_pts[i, :] - dst_pts[j, :]))

    P_mat = np.hstack((dst_pts, np.ones((p, 1))))
    
    K_P_Pt_mat_row1 = np.hstack((K_mat, P_mat))
    K_P_Pt_mat_row2 = np.hstack((P_mat.T, np.zeros((3, 3))))
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
        f1 += x_param[i] * U(np.linalg.norm(dst_pts[i, :] - [img_warp_row, img_warp_col]))
        f2 += y_param[i] * U(np.linalg.norm(dst_pts[i, :] - [img_warp_row, img_warp_col]))
    x = a1_x + ax_x * img_warp_row + ay_x * img_warp_col + f1
    y = a1_y + ax_y * img_warp_row + ay_y * img_warp_col + f2
    return x, y

def thin_plate_spline_warping(src, dst, src_pts, dst_pts, dst_hull):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    p = len(src_pts)
    r = cv2.boundingRect(np.float32([dst_pts]))
    mask = np.zeros((r[3], r[2], 3), np.float32)

    points2_t = []

    for i in range(len(dst_hull)):
        points2_t.append(((dst_hull[i][0]-r[0]),(dst_hull[i][1]-r[1])))

    cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0)

    warped_img = np.copy(mask)

    est_params_x = tps_model(src_pts[:, 0], dst_pts)
    est_params_y = tps_model(src_pts[:, 1], dst_pts)

    a1_x = est_params_x[p+2]
    ay_x = est_params_x[p+1]
    ax_x = est_params_x[p]

    a1_y = est_params_y[p+2]
    ay_y = est_params_y[p+1]
    ax_y = est_params_y[p]
    for i in range(warped_img.shape[1]):
        for j in range(warped_img.shape[0]):
            x, y = f(est_params_x, est_params_y, p, dst_pts, i + r[0], j + r[1])
            x = min(max(int(x), 0), src.shape[1]-1)
            y = min(max(int(y), 0), src.shape[0]-1)
            warped_img[j, i] = src[y, x, :] 
    warped_img = warped_img * mask
    dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( (1.0, 1.0, 1.0) - mask )
    dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = dst[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] + warped_img
    
    return dst

def traditional(src, dst, src_face, dst_face, method):
    dst_copy = np.copy(dst)
    dst_face = np.array(dst_face)
    hull = cv2.convexHull(np.float32([dst_face]), returnPoints=False)
    src_hull = []
    dst_hull = []
    for h in hull:
        src_hull.append(src_face[int(h)])
        dst_hull.append(dst_face[int(h)])

    if method == 'tri':
        warped_img = triangulation_warping(src, dst, src_face, dst_face, dst_hull)
        cv2.imshow('w', warped_img)
        cv2.waitKey(0)
    elif method == 'tps':
        warped_img = thin_plate_spline_warping(src, dst_copy, src_face, dst_face, dst_hull)
        # cv2.imshow('w', warped_img)
        # cv2.waitKey(0)
    output = blend(warped_img, dst, dst_hull)
    # cv2.destroyAllWindows()

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
    parser.add_argument('--method', default='tps', type=str, help='tri, tps')
    Args = parser.parse_args()
    method = Args.method

    video_path = './TestSet/Test1.mp4'
    face_img_path = './TestSet/Rambo.jpg'
    cap = cv2.VideoCapture(video_path)
    face_img = cv2.imread(face_img_path)
    (x, y, w, h), face1_pts = facial_landmark(face_img)
    face1_crop = face_img[y:y+h, x:x+w]
    if (cap.isOpened()== False): 
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            (x, y, w, h), face2_pts = facial_landmark(frame)
            face2_crop = frame[y:y+h, x:x+w]
            res = traditional(face_img, frame, face1_pts, face2_pts, method)
            cv2.imshow('r', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()