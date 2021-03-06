import numpy as np
import cv2

def U_func(r):
    res = (r**2) * np.log(r**2) # if r >= 0.0001 else 0
    return np.nan_to_num(res)

def tps_model(src_1d, dst_pts):
    p = len(dst_pts)
    K_mat = np.zeros((p, p), np.float32) # size: p x p
    P_mat = np.zeros((p, 3), np.float32) # size: p x 3
    for j in range(p):
        r = np.linalg.norm(dst_pts - dst_pts[j, :], axis=1)
        K_mat[:, j] = U_func(r)

    P_mat = np.hstack((dst_pts, np.ones((p, 1))))
    
    K_P_Pt_mat_row1 = np.hstack((K_mat, P_mat))
    K_P_Pt_mat_row2 = np.hstack((P_mat.T, np.zeros((3, 3))))
    K_P_Pt_mat = np.vstack((K_P_Pt_mat_row1, K_P_Pt_mat_row2))
    lamda = 0.0001
    A_mat = K_P_Pt_mat + lamda * np.identity((p + 3))
    v_o = np.concatenate((src_1d, [0, 0, 0]))
    est_params = np.matmul(np.linalg.inv(A_mat), v_o)
    return est_params

def f(x_param, y_param, p, dst_pts, img_warp_row, img_warp_col):
    ax_x, ay_x, a1_x = x_param[p], x_param[p+1], x_param[p+2]
    ax_y, ay_y, a1_y = y_param[p], y_param[p+1], y_param[p+2]
    f1, f2 = 0, 0
    for i in range(p):
        f1 += x_param[i] * U_func(np.linalg.norm(dst_pts[i, :] - [img_warp_row, img_warp_col]))
        f2 += y_param[i] * U_func(np.linalg.norm(dst_pts[i, :] - [img_warp_row, img_warp_col]))
    x = a1_x + ax_x * img_warp_row + ay_x * img_warp_col + f1
    y = a1_y + ax_y * img_warp_row + ay_y * img_warp_col + f2
    return x, y

def thin_plate_spline_warping(src, dst, src_pts, dst_pts, dst_hull):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    p = len(src_pts)
    (x, y, w, h) = cv2.boundingRect(np.float32([dst_pts]))
    mask = np.zeros((h, w, 3), np.float32)

    pts2_t = []

    for dh in dst_hull:
        pts2_t.append(((dh[0]-x),(dh[1]-y)))

    cv2.fillConvexPoly(mask, np.int32(pts2_t), (1.0, 1.0, 1.0), 16, 0)


    est_params_x = tps_model(src_pts[:, 0], dst_pts) # size: 68+3
    est_params_y = tps_model(src_pts[:, 1], dst_pts) # size: 68+3

    est_params_x = est_params_x.reshape(-1,1) # size: 71 x 1
    est_params_y = est_params_y.reshape(-1,1) # size: 71 x 1

    Xi, Yi = np.indices((dst.shape[1], dst.shape[0])) 
    warped_pts = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size))).T
    
    axx, ayx, a1x = est_params_x[p], est_params_x[p + 1], est_params_x[p + 2]
    axy, ayy, a1y = est_params_y[p], est_params_y[p + 1], est_params_y[p + 2]

    A = np.array([[axx, axy], [ayx, ayy], [a1x, a1y]]).reshape(3,2)
    actual_pts = np.dot(warped_pts, A) 

    warped_pts_x = warped_pts[:,0].reshape(-1,1)
    warped_pts_y = warped_pts[:, 1].reshape(-1,1)
    
    dst_pts_x = dst_pts[:,0].reshape(-1,1)
    dst_pts_y = dst_pts[:, 1].reshape(-1,1)

    ax, bx = np.meshgrid(dst_pts_x, warped_pts_x)
    ay, by = np.meshgrid(dst_pts_y, warped_pts_y)
    # calculate the L1 norm
    R = np.sqrt((ax - bx)**2 + (ay - by)**2) 

    U = (R**2) * np.log(R**2)
    U[R == 0] = 0

    est_params_x = est_params_x[:68, 0].T
    est_params_y = est_params_y[:68, 0].T
    
    fx = est_params_x * U 
    fy = est_params_y * U

    fx_sum = np.sum(fx, axis = 1).reshape(-1,1)
    fy_sum = np.sum(fy, axis = 1).reshape(-1,1)

    actual_pts = actual_pts + np.hstack((fx_sum, fy_sum))
    X = actual_pts[:, 0].astype(int)
    Y = actual_pts[:, 1].astype(int)
    X[X >= src.shape[1]] = src.shape[1] - 1
    Y[Y >= src.shape[0]] = src.shape[0] - 1
    X[X < 0] = 0
    Y[Y < 0] = 0
    warped_img = np.zeros(dst.shape)
    warped_img[Yi.ravel(), Xi.ravel()] = src[Y, X]
    
    # Slow warping result
    # warped_img = np.copy(mask)
    # for i in range(warped_img.shape[1]):
    #     for j in range(warped_img.shape[0]):
    #         a, b = f(est_params_x, est_params_y, p, dst_pts, i + x, j + y)
    #         a = min(max(int(a), 0), src.shape[1]-1)
    #         b = min(max(int(b), 0), src.shape[0]-1)
    #         warped_img[j, i] = src[b, a, :] 

    # warped_img = warped_img * mask
    # dst_rect_area = dst[y:y+h, x:x+w]
    # dst_rect_area = dst_rect_area * ((1.0, 1.0, 1.0) - mask)
    # dst_rect_area = dst_rect_area + warped_img
    # dst[y:y+h, x:x+w] = dst_rect_area

    return warped_img