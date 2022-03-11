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
    for f in faces:
        subdiv.insert(tuple(f))
    triangle_list = subdiv.getTriangleList()

    face_pts = dict()
    for i, j in enumerate(faces):
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

    t = dst_tri

    dst_rect = cv2.boundingRect(np.float32([t]))

    xleft = dst_rect[0]
    xright = dst_rect[0] + dst_rect[2]
    ytop = dst_rect[1]
    ybottom = dst_rect[1] + dst_rect[3]

    dst_matrix = np.linalg.inv([[t[0][0],t[1][0],t[2][0]],[t[0][1],t[1][1],t[2][1]],[1,1,1]])     

    grid = np.mgrid[xleft:xright, ytop:ybottom].reshape(2,-1)
    #grid 2xN
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    #grid 3xN
    barycen_cord = np.dot(dst_matrix, grid)

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

    src_matrix = np.matrix([[src_tri[0][0],src_tri[1][0],src_tri[2][0]],
                            [src_tri[0][1],src_tri[1][1],src_tri[2][1]],[1,1,1]])
    pts = np.matmul(src_matrix,barycen_cord)
    
    src_x = pts[0,:]/pts[2,:]
    src_y = pts[1,:]/pts[2,:]
    
    # copy back the value of pixel at (x_A, y_A) to the target location
    xs = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
    ys = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)
    blue, green, red = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    b_eq = interp2d(xs, ys, blue, kind='cubic')
    g_eq = interp2d(xs, ys, green, kind='cubic')
    r_eq = interp2d(xs, ys, red, kind='cubic')
    dst= np.zeros((size[1], size[0], 3), np.uint8)
    for ind, (x, y) in enumerate(zip(src_x.flat, src_y.flat)):
        b = b_eq(x, y)[0]
        g = g_eq(x, y)[0]
        r = r_eq(x, y)[0]
        dst[int(dst_y[ind-2]), int(dst_x[ind-2])] = (b, g, r)
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
    # print("Mask shape = "+str(mask.shape))

    cv2.fillConvexPoly(mask, np.int32(del_tri_rect2), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[b_rect1[1]:b_rect1[1] + b_rect1[3], b_rect1[0]:b_rect1[0] + b_rect1[2]]
    img2Rect = np.zeros((b_rect2[3], b_rect2[2]), dtype=img1Rect.dtype)

    size = (b_rect2[2], b_rect2[3])

    # img2Rect = triangulationWarping(img1Rect, t1Rect, t2Rect, size)
    img2Rect = triangulation_model(img1Rect, del_tri_rect1, del_tri_rect2, size)
    # print("img2rect shape =  "+str(img2Rect.shape))

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