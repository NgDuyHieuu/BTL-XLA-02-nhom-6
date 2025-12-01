import cv2
import numpy as np
import matplotlib.pyplot as plt

# HÀM GHÉP 2 ẢNH
def panorama(img1, img2):
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

# HÀM HIỂN THỊ NHIỀU ẢNH
def show_images(images, titles=None, figsize=(16,6)):
    n = len(images)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(1, n)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles or [""]*n):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    file_list = ['image3.jpg', 'image4.jpg', 'image6.jpg']
    # Đọc ảnh
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in file_list]

    # Resize cho cùng chiều cao
    h_min = min(img.shape[0] for img in imgs)
    imgs = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min)) for img in imgs]

    # Hiển thị 3 ảnh gốc
    show_images(imgs, titles=["Ảnh 1", "Ảnh 2", "Ảnh 3"])
