import cv2

src_image = cv2.imread('/mnt/cephfs/dataset/Face/EG3D_GEN_W_0.3/000044/000044_s.jpg')
M = cv2.getRotationMatrix2D((256,256), 5, 1)
rotate_30 = cv2.warpAffine(src_image,M, (512,512)) 
cv2.imwrite("samples/000044_s_crop.jpg", rotate_30)