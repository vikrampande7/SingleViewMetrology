# Importing all the necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import math

# 1) ----- IMAGE ACQUSITION AND ANNOTATION -----
# Reading the image
img_ip = cv2.imread('IMG-2632.png')

#downscaling the image
w_ip, h_ip, d = img_ip.shape
down_width = int(w_ip/3)
down_height = int(h_ip/4)
down_points = (down_width, down_height)
img = cv2.resize(img_ip, down_points, interpolation= cv2.INTER_LINEAR)
img_copy = img.copy()

# --------- ANNOTATION ---------

# Plotting the co-ordinates and lines
# Total points taken are 7 so that we can plot two lines
c1 = [531,765]
c2 = [762,533]
c3 = [253,460]
c4 = [535,520]
c5 = [795,302]
c6 = [219,237]
c7 = [466,131]

# 1) X axis - Blue 
cv2.line(img,c1,c2,(0,0,255),4)
cv2.line(img,c4,c5,(0,0,255),4)
cv2.line(img,c6,c7,(0,0,255),4)
# 2) Y axis - Green
cv2.line(img,c1,c3,(0,255,0),4)
cv2.line(img,c4,c6,(0,255,0),4)
cv2.line(img,c5,c7,(0,255,0),4)
# 3) Z axis - Red
cv2.line(img,c1,c4,(255,0,0),4)
cv2.line(img,c3,c6,(255,0,0),4)
cv2.line(img,c2,c5,(255,0,0),4)

# plt.imshow(img)
# plt.show()
# cv2.imwrite("annotated.png",img)


# 2) ----- COMPUTE VANISH POINTS -----
# adding 1 as third element

# 1) Blue X axis
v1_x1 = [c1[0], c1[1], 1]
v1_x2 = [c2[0], c2[1], 1]
v2_x1 = [c4[0], c4[1], 1]
v2_x2 = [c5[0], c5[1], 1]

b1_x1,b1_x2,b1_x3 = np.cross(v1_x1,v1_x2)
b2_x1,b2_x2,b2_x3 = np.cross(v2_x1,v2_x2)
Vx = np.cross([b1_x1,b1_x2,b1_x3],[b2_x1,b2_x2,b2_x3])
Vx = Vx/Vx[2]


# 2) Green Y axis
v1_y1 = [c1[0], c1[1], 1]
v1_y2 = [c3[0], c3[1], 1]
v2_y1 = [c4[0], c4[1], 1]
v2_y2 = [c6[0], c6[1], 1]

g1_y1,g1_y2,g1_y3 = np.cross(v1_y1,v1_y2)
g2_y1,g2_y2,g2_y3 = np.cross(v2_y1,v2_y2)
Vy = np.cross([g1_y1,g1_y2,g1_y3],[g2_y1,g2_y2,g2_y3])
Vy = Vy/Vy[2]


# 3) Red Z axis
v1_z1 = [c1[0], c1[1], 1]
v1_z2 = [c4[0], c4[1], 1]
v2_z1 = [c2[0], c2[1], 1]
v2_z2 = [c5[0], c5[1], 1]

r1_z1,r1_z2,r1_z3 = np.cross(v1_z1,v1_z2)
r2_z1,r2_z2,r2_z3 = np.cross(v2_z1,v2_z2)
Vz = np.cross([r1_z1,r1_z2,r1_z3],[r2_z1,r2_z2,r2_z3])
Vz = Vz/Vz[2]

# 3) ----- CONSTRUCTING PROJECTION MATRIX AND HOMOGRAPH MATRIX -----
#Taking one reference point and world coordinates
w0 = [c1[0], c1[1], 1]
ref_x = [c2[0], c2[1], 1]
ref_y = [c3[0], c3[1], 1]
ref_z = [c4[0], c4[1], 1]
ref_x = np.array([ref_x])
ref_y = np.array([ref_y])
ref_z = np.array([ref_z])

distance_x = np.sqrt(np.sum(np.square(ref_x - w0)))   
distance_y = np.sqrt(np.sum(np.square(ref_y - w0)))   
distance_z = np.sqrt(np.sum(np.square(ref_z - w0)))   

# Converting all the required elements to array
Vx = np.array(Vx)
Vy = np.array(Vy)
Vz = np.array(Vz)
w0 = np.array(w0)
ref_x = np.array(ref_x)
ref_y = np.array(ref_y)
ref_z = np.array(ref_z)


#Getting a Scaling Factor
ax,resid,rank,s = np.linalg.lstsq( (Vx-ref_x).T , (ref_x - w0).T, rcond=None )
ax = ax[0][0]/distance_x

ay,resid,rank,s = np.linalg.lstsq( (Vy-ref_y).T , (ref_y - w0).T, rcond=None )
ay = ay[0][0]/distance_y

az,resid,rank,s = np.linalg.lstsq( (Vz-ref_z).T , (ref_z - w0).T, rcond=None )
az = az[0][0]/distance_z

# Constructing Projection Matrix
pmx = ax*Vx
pmy = ay*Vy
pmz = az*Vz
# --------- PROJECTION MATRIX ---------
proj_mat = np.empty([3,4])
proj_mat[:,0] = pmx
proj_mat[:,1] = pmy
proj_mat[:,2] = pmz
proj_mat[:,3] = w0

# print("Projection Matrix\n",proj_mat)

# Constructing Homography Matrix
Hxy = np.zeros((3,3))
Hyz = np.zeros((3,3))
Hzx = np.zeros((3,3))

Hxy[:,0] = pmx
Hxy[:,1] = pmy
Hxy[:,2] = w0

Hyz[:,0] = pmy
Hyz[:,1] = pmz
Hyz[:,2] = w0

Hzx[:,0] = pmx
Hzx[:,1] = pmz
Hzx[:,2] = w0

# Adjusting the image to make it visible
Hxy[0,2] = Hxy[0,2] + 30
Hxy[1,2] = Hxy[1,2] 

Hyz[0,2] = Hyz[0,2] + 300
Hyz[1,2] = Hyz[1,2] + 600

Hzx[0,2] = Hzx[0,2] + -300
Hzx[1,2] = Hzx[1,2] + 400


# 4) ----- GETTING THE TEXTURE MAPS -----
w,h,temp = img.shape
# Getting Texture Maps
TM_xy = cv2.warpPerspective(img_copy,Hxy,(w,h),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Txy",TM_xy)
cv2.imwrite("XY_plane.png",TM_xy)
cv2.waitKey(0)


TM_yz = cv2.warpPerspective(img_copy,Hyz,(w,h),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Tyz",TM_yz)
cv2.imwrite("YZ_plane.png",TM_yz)
cv2.waitKey(0)


TM_zx = cv2.warpPerspective(img_copy,Hzx,(w,h),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Txz",TM_zx)
cv2.imwrite("ZX_plane.png",TM_zx)
cv2.waitKey(0)

cv2.destroyAllWindows()