import cv2
import dlib
import argparse
import numpy as np
from scipy.interpolate import interp2d
from imutils import face_utils

def visualize_img(img):
    cv2.imshow('t', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

# Check if a point is inside a rectangle
def rect_contains(box, pt) :
    x,y =  pt
    x_min,y_min,width,height = box
    x_max,y_max = x_min + width, y_min + height
    
    if (x > x_min) and (x < x_max) and (y > y_min) and (y < y_max):
        return True
    else:
        return False

def delaunay_triangle(img, points):
    box = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(box) 
    # insert points into subdiv
    for p in points:
        subdiv.insert(tuple(p))
    triangle_list = subdiv.getTriangleList()

    face_pts = dict()
    for i, j in enumerate(points):
        k = tuple(j)
        face_pts[k] = i

    res = []
    for t in triangle_list:
        pt_1 = (t[0], t[1])
        pt_2 = (t[2], t[3])
        pt_3 = (t[4], t[5])
        pts = [pt_1, pt_2, pt_3]

        if rect_contains(box, pt_1) and rect_contains(box, pt_2) and rect_contains(box, pt_3):
            indices = []
            for p in pts:
                idx = face_pts.get(tuple(p), False)
                if idx:
                    indices.append(idx)
            # three points form a delaunay triangle        
            if len(indices) == 3:
                res.append(np.array(indices))
    return res

def triangulation_model(src, src_tri, dst_tri, size):
    # compute Barycentric coordinates for each triangle in the destination
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    x_left = dst_rect[0]
    x_right = dst_rect[0] + dst_rect[2]
    y_top = dst_rect[1]
    y_bottom = dst_rect[1] + dst_rect[3]
    tri_coord = np.mgrid[x_left:x_right, y_top:y_bottom].reshape(2,-1)
    B_mat = [[dst_tri[0][0], dst_tri[1][0], dst_tri[2][0]],
            [dst_tri[0][1], dst_tri[1][1], dst_tri[2][1]], [1, 1, 1]]
    B_mat_inv = np.linalg.inv(B_mat)
    ones = np.ones((1, tri_coord.shape[1]))
    tri_coord = np.vstack((tri_coord, ones))
    barycen_cord = np.dot(B_mat_inv, tri_coord)
    
    epsilon = 0.1
    t =[]
    b = np.all(barycen_cord>-epsilon, axis=0)
    a = np.all(barycen_cord<1+epsilon, axis=0)
    for i in range(len(a)):
        t.append(a[i] and b[i])
    dst_y = []
    dst_x = []
    for i in range(len(t)):
        if(t[i]):
            dst_y.append(i%dst_rect[3])
            dst_x.append(i/dst_rect[3])

    barycen_cord = barycen_cord[:,np.all(-epsilon<barycen_cord, axis=0)]
    barycen_cord = barycen_cord[:,np.all(barycen_cord<1+epsilon, axis=0)]

    src_matrix = np.array([[src_tri[0][0],src_tri[1][0],src_tri[2][0]],
                            [src_tri[0][1],src_tri[1][1],src_tri[2][1]],[1,1,1]])
    pts = src_matrix @ barycen_cord
    
    src_x = pts[0,:]/pts[2,:]
    src_y = pts[1,:]/pts[2,:]

    # copy back the value of pixel at (x_A, y_A) to the target location
    dst= np.zeros((size[1], size[0], 3), np.uint8)
    xs = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
    ys = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)
    blue, green, red = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    b_eq = interp2d(xs, ys, blue, kind='cubic')
    g_eq = interp2d(xs, ys, green, kind='cubic')
    r_eq = interp2d(xs, ys, red, kind='cubic')
    for ind, (x, y) in enumerate(zip(src_x.flat, src_y.flat)):
        b = b_eq(x, y)[0]
        g = g_eq(x, y)[0]
        r = r_eq(x, y)[0]
        dst[int(dst_y[ind]), int(dst_x[ind])] = (b, g, r)
    return dst

def warpTriangle(img1, img2, del_tri1, del_tri2):
    # Find bounding rectangle for each triangle
    b_rect1 = cv2.boundingRect(np.float32([del_tri1]))
    b_rect2 = cv2.boundingRect(np.float32([del_tri2]))
    # Offset points by left top corner of the respective rectangles
    del_tri_rect1 = []
    del_tri_rect2 = []

    for r1, r2 in zip(del_tri1, del_tri2):
        del_tri_rect1.append((r1[0] - b_rect1[0], (r1[1] - b_rect1[1])))
        del_tri_rect2.append((r2[0] - b_rect2[0], (r2[1] - b_rect2[1])))

    # Get mask by filling triangle
    mask = np.zeros((b_rect2[3], b_rect2[2], 3), np.float32)

    cv2.fillConvexPoly(mask, np.int32(del_tri_rect2), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[b_rect1[1]:b_rect1[1] + b_rect1[3], b_rect1[0]:b_rect1[0] + b_rect1[2]]
    img2Rect = np.zeros((b_rect2[3], b_rect2[2]), img1Rect.dtype)

    size = (b_rect2[2], b_rect2[3])

    img2Rect = triangulation_model(img1Rect, del_tri_rect1, del_tri_rect2, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[b_rect2[1]:b_rect2[1]+b_rect2[3], b_rect2[0]:b_rect2[0]+b_rect2[2]] = img2[b_rect2[1]:b_rect2[1]+b_rect2[3], b_rect2[0]:b_rect2[0]+b_rect2[2]] * ((1.0, 1.0, 1.0) - mask)

    img2[b_rect2[1]:b_rect2[1]+b_rect2[3], b_rect2[0]:b_rect2[0]+b_rect2[2]] = img2[b_rect2[1]
        :b_rect2[1]+b_rect2[3], b_rect2[0]:b_rect2[0]+b_rect2[2]] + img2Rect

def triangulation_warping(src, dst, dst_copy, src_hull, dst_hull):
    # delaunay triangulation for convex hull
    del_tri = delaunay_triangle(dst, dst_hull)
    # apply affine transformation to delaunay triangles
    for dt in del_tri:
        del_tri1 = []
        del_tri2 = []
        
        for i in range(3):
            del_tri1.append(src_hull[dt[i]])
            del_tri2.append(dst_hull[dt[i]])
        warpTriangle(src, dst_copy, del_tri1, del_tri2)
    return dst_copy

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
    hull_idx = cv2.convexHull(np.array(dst_face), returnPoints=False)
    src_hull = []
    dst_hull = []
    for h in hull_idx:
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
    parser.add_argument('--method', default='tps', type=str, help='tri, tps')
    Args = parser.parse_args()
    method = Args.method

    video_path = './TestSet/Test1.mp4'
    face_img_path = './TestSet/Rambo.jpg'
    cap = cv2.VideoCapture(video_path)
    face_img = cv2.imread(face_img_path)
    _, face1_pts = facial_landmark(face_img)
    if (cap.isOpened()== False): 
        print("Error opening video file")
    count = 0
    while(cap.isOpened()):
        count += 1
        ret, frame = cap.read()
        if ret == True:
            num_face2, face2_pts = facial_landmark(frame)
            if(num_face2 == 0):
                continue
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