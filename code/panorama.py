import cv2
import numpy as np
import matplotlib.pyplot as plt

def panorama2(img1, img2):
    # Chuyển ảnh sang ảnh xám để xử lý đặc trưng SIFT
    scr_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    tar_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Tạo bộ phát hiện SIFT
    Sift_detect = cv2.xfeatures2d.SIFT_create()

    # Phát hiện keypoints và descriptors từ ảnh xám
    
    k1, d1 = Sift_detect.detectAndCompute(scr_gray, None)
    k2, d2 = Sift_detect.detectAndCompute(tar_gray, None)

    # Dùng BFMatcher với KNN (k=2) để tìm các cặp mô tả tương đồng

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(d1, d2, k=2)

    # Áp dụng Lowe's Ratio Test

    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)


    # Vẽ đường nối giữa các điểm khớp
    img_soKhop = cv2.drawMatches(img1, k1, img2, k2, good_matches, None, flags=2)

    # Lấy tọa độ keypoints tương ứng từ các match


    keypoint_1 = np.float32([kp.pt for kp in k1])
    keypoint_2 = np.float32([kp.pt for kp in k2])
    des_1 = np.float32([keypoint_1[m.queryIdx] for m in good_matches])
    des_2 = np.float32([keypoint_2[m.trainIdx] for m in good_matches])

    #Tính homography từ ảnh 2 về ảnh 1 bằng RANSAC 


    H, _ = cv2.findHomography(des_2, des_1, cv2.RANSAC)

    # Lấy kích thước ảnh

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Biến đổi ảnh 2 theo homography để ghép vào ảnh 1

    img_result = cv2.warpPerspective(img2, H, (w1 + w2, h2))  # tạo ảnh kết quả đủ lớn để chứa cả 2 ảnh
    img_result[0:h1, 0:w1] = img1  # chèn ảnh 1 vào kết quả

    # Xử lý cắt phần dư (vùng đen không có nội dung bên phải)

    gray = cv2.cvtColor(img_result, cv2.COLOR_RGB2GRAY)
    cols = np.where(gray.sum(axis=0) > 0)[0]  
    if len(cols) > 0:
        right = cols[-1]  
        img_result = img_result[:, :right+1] 

    return img_soKhop, img_result

if __name__ == "__main__":
    # Đọc 2 ảnh từ file
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')

    # Resize ảnh cho cùng chiều cao để dễ ghép
    h_min = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h_min / img1.shape[0]), h_min))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h_min / img2.shape[0]), h_min))

    # Chuyển ảnh từ BGR (mặc định OpenCV) sang RGB (để hiển thị bằng matplotlib)
    scr_imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    tar_imgRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Gọi hàm ghép panorama
    img_soKhop, img_Panorama = panorama2(scr_imgRGB, tar_imgRGB)

    # Hiển thị 2 ảnh gốc
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.imshow(scr_imgRGB)
    ax1.set_title('Ảnh 1')
    ax1.axis('off')

    ax2.imshow(tar_imgRGB)
    ax2.set_title('Ảnh 2')
    ax2.axis('off')
    plt.show()

    # Hiển thị ảnh so khớp và ảnh panorama
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(2, 1)

    ax1.imshow(img_soKhop)
    ax1.set_title('Ảnh so khớp ảnh 1 và ảnh 2')
    ax1.axis('off')

    ax2.imshow(img_Panorama)
    ax2.set_title('Ảnh Panorama tạo từ ảnh 1 và ảnh 2')
    ax2.axis('off')
    plt.show()
