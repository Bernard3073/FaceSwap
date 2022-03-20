import cv2
import numpy as np
from scipy.interpolate import interp2d

# Check if a point is inside a rectangle
def rect_contains(box, pt) :
    x,y =  pt
    x_min,y_min,width,height = box
    x_max,y_max = x_min + width, y_min + height
    
    if (x > x_min) and (x < x_max) and (y > y_min) and (y < y_max):
        return True
    else:
        return False

def delaunay_triangle(img, faces):
    box = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(box) 
    # insert points into subdiv
    face_pts = dict()
    for i, f in enumerate(faces):
        subdiv.insert(tuple(f))
        face_pts[tuple(f)] = i
    triangle_list = subdiv.getTriangleList()

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

def triangulation_model(src, src_tri, dst_tri, h2, w2):

    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    x_left = dst_rect[0]
    x_right = dst_rect[0] + dst_rect[2]
    y_top = dst_rect[1]
    y_bottom = dst_rect[1] + dst_rect[3]
    # for each triangle in the destination face, compute the Barycentric coordinate
    B_mat_inv = np.linalg.inv([[dst_tri[0][0], dst_tri[1][0], dst_tri[2][0]], 
                            [dst_tri[0][1], dst_tri[1][1], dst_tri[2][1]],
                            [1, 1, 1]])     

    dst_tri_x, dst_tri_y = np.mgrid[x_left:x_right, y_top:y_bottom].reshape(2,-1)
    ones = np.ones((1, dst_tri_x.shape[0]))
    dst_tri_coord = np.vstack((dst_tri_x, dst_tri_y, ones))
    barycen_cord = np.matmul(B_mat_inv, dst_tri_coord)

    epsilon = 0.1
    t =[]
    lower_bound = np.all(barycen_cord > -epsilon, axis=0)
    upper_bound = np.all(barycen_cord < 1+epsilon, axis=0)
    # find index for the target location
    dst_y = []
    dst_x = []
    for i, (l, u) in enumerate(zip(lower_bound, upper_bound)):
        if l and u:
            dst_y.append(i % dst_rect[3])
            dst_x.append(i / dst_rect[3])

    # make the Barycentric coordinate lies inside the triangle
    barycen_cord = barycen_cord[:, np.all(barycen_cord > -epsilon, axis=0)]
    barycen_cord = barycen_cord[:, np.all(barycen_cord < 1+epsilon, axis=0)]
    # compute the corresponding pixel position in the source image
    A_mat = np.array([[src_tri[0][0], src_tri[1][0], src_tri[2][0]],
                      [src_tri[0][1], src_tri[1][1], src_tri[2][1]],
                      [1, 1, 1]])
    pts = np.matmul(A_mat, barycen_cord)
    # convert to homogeneous coordinates
    src_x = pts[0,:] / pts[2,:]
    src_y = pts[1,:] / pts[2,:]
    
    # copy back the value of pixel at (x_A, y_A) to the target location
    xs = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
    ys = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)
    blue, green, red = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    b_eq = interp2d(xs, ys, blue, kind='cubic')
    g_eq = interp2d(xs, ys, green, kind='cubic')
    r_eq = interp2d(xs, ys, red, kind='cubic')
    dst= np.zeros((h2, w2, 3), np.uint8)
    for ind, (x, y) in enumerate(zip(src_x.flat, src_y.flat)):
        b = b_eq(x, y)[0]
        g = g_eq(x, y)[0]
        r = r_eq(x, y)[0]
        dst[int(dst_y[ind]), int(dst_x[ind])] = (b, g, r)
    return dst

def warpTriangle(img1, img2, del_tri1, del_tri2):
    # Find bounding rectangle for each triangle
    (x1, y1, w1, h1) = cv2.boundingRect(np.float32([del_tri1]))
    (x2, y2, w2, h2) = cv2.boundingRect(np.float32([del_tri2]))

    del_tri_rect1 = []
    del_tri_rect2 = []

    for r1, r2 in zip(del_tri1, del_tri2):
        del_tri_rect1.append((r1[0] - x1, (r1[1] - y1)))
        del_tri_rect2.append((r2[0] - x2, (r2[1] - y2)))
    # Get mask by filling triangle
    mask = np.zeros((h2, w2, 3), np.float32)

    cv2.fillConvexPoly(mask, np.int32(del_tri_rect2), (1.0, 1.0, 1.0), 16, 0)

    # cropped triangles in img1
    img1Rect = img1[y1:y1 + h1, x1:x1 + w1]

    img2Rect = np.zeros((h2, w2), dtype=img1Rect.dtype)


    img2Rect = triangulation_model(img1Rect, del_tri_rect1, del_tri_rect2, h2, w2)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2_rect_area = img2[y2:y2+h2, x2:x2+w2]
    img2_rect_area = img2_rect_area * ((1.0, 1.0, 1.0) - mask)
    img2_rect_area = img2_rect_area + img2Rect
    img2[y2:y2+h2, x2:x2+w2] = img2_rect_area

def triangulation_warping(src, dst, dst_copy, src_hull, dst_hull):
    # delaunay triangulation for convex hull
    del_tri = delaunay_triangle(dst, dst_hull)
    # apply affine transformation to delaunay triangles
    for (dt1, dt2, dt3) in del_tri:
        del_tri1 = [src_hull[dt1], src_hull[dt2], src_hull[dt3]]
        del_tri2 = [dst_hull[dt1], dst_hull[dt2], dst_hull[dt3]]
        warpTriangle(src, dst_copy, del_tri1, del_tri2)
    return dst_copy