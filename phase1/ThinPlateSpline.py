import numpy as np
import cv2

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
    (x, y, w, h) = cv2.boundingRect(np.float32([dst_pts]))
    mask = np.zeros((h, w, 3), np.float32)

    points2_t = []

    for dh in dst_hull:
        points2_t.append(((dh[0]-x),(dh[1]-y)))

    cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0)

    warped_img = np.copy(mask)

    est_params_x = tps_model(src_pts[:, 0], dst_pts)
    est_params_y = tps_model(src_pts[:, 1], dst_pts)

    for i in range(warped_img.shape[1]):
        for j in range(warped_img.shape[0]):
            x, y = f(est_params_x, est_params_y, p, dst_pts, i + x, j + y)
            x = min(max(int(x), 0), src.shape[1]-1)
            y = min(max(int(y), 0), src.shape[0]-1)
            warped_img[j, i] = src[y, x, :] 
            
    warped_img = warped_img * mask
    dst_rect_area = dst[y:y+h, x:x+w]
    dst_rect_area = dst_rect_area * ((1.0, 1.0, 1.0) - mask)
    dst_rect_area = dst_rect_area + warped_img
    dst[y:y+h, x:x+w] = dst_rect_area
    
    return dst