import cv2
import numpy as np
import matplotlib.pyplot as plt

def panorama(img1, img2):
    """
    Ghép ảnh panorama tự động phát hiện thứ tự đúng
    """
    scr_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    tar_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    Sift_detect = cv2.xfeatures2d.SIFT_create()
    k1, d1 = Sift_detect.detectAndCompute(scr_gray, None)
    k2, d2 = Sift_detect.detectAndCompute(tar_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:200]
    
    keypoint_1 = np.float32([kp.pt for kp in k1])
    keypoint_2 = np.float32([kp.pt for kp in k2])
    pts1 = np.float32([keypoint_1[m.queryIdx] for m in matches])
    pts2 = np.float32([keypoint_2[m.trainIdx] for m in matches])
    
    # Phân tích vị trí keypoints để xác định thứ tự
    # Nếu ảnh 1 ở bên trái, các điểm matching ở ảnh 1 sẽ ở bên phải
    # và các điểm matching ở ảnh 2 sẽ ở bên trái
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Tính trung bình vị trí X của các điểm matching
    avg_x1 = np.mean(pts1[:, 0])
    avg_x2 = np.mean(pts2[:, 0])
    
    # Tỷ lệ vị trí so với chiều rộng ảnh
    ratio1 = avg_x1 / w1  # Tỷ lệ trong ảnh 1
    ratio2 = avg_x2 / w2  # Tỷ lệ trong ảnh 2
    
    # Nếu ảnh 1 ở trái: ratio1 cao (điểm ở bên phải ảnh 1), ratio2 thấp (điểm ở bên trái ảnh 2)
    # Nếu ảnh 1 ở phải: ratio1 thấp, ratio2 cao
    is_img1_left = (ratio1 > 0.5 and ratio2 < 0.5) or (ratio1 > ratio2 + 0.1)
    
    # Thử cả 2 cách và chọn kết quả tốt hơn
    results = []
    
    # Cách 1: img1 trái, img2 phải
    try:
        H1, mask1 = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H1 is not None:
            inliers1 = np.sum(mask1) if mask1 is not None else 0
            # Kiểm tra homography hợp lệ (không bị biến dạng quá mức)
            if inliers1 > 10:  # Ít nhất 10 inliers
                img_result1 = cv2.warpPerspective(img2, H1, (w1 + w2, max(h1, h2)))
                img_result1[0:h1, 0:w1] = img1
                # Đánh giá chất lượng: tỷ lệ diện tích sử dụng
                gray1 = cv2.cvtColor(img_result1, cv2.COLOR_RGB2GRAY)
                used_area1 = np.sum(gray1 > 0) / (gray1.shape[0] * gray1.shape[1])
                score1 = inliers1 * 0.7 + used_area1 * 100  # Trọng số cho inliers cao hơn
                results.append((score1, img_result1, H1, mask1, True))
    except:
        pass
    
    # Cách 2: img2 trái, img1 phải (đảo ngược)
    try:
        H2, mask2 = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H2 is not None:
            inliers2 = np.sum(mask2) if mask2 is not None else 0
            if inliers2 > 10:
                img_result2 = cv2.warpPerspective(img1, H2, (w1 + w2, max(h1, h2)))
                img_result2[0:h2, 0:w2] = img2
                gray2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2GRAY)
                used_area2 = np.sum(gray2 > 0) / (gray2.shape[0] * gray2.shape[1])
                score2 = inliers2 * 0.7 + used_area2 * 100
                results.append((score2, img_result2, H2, mask2, False))
    except:
        pass
    
    # Chọn kết quả tốt nhất
    if len(results) == 0:
        # Fallback về cách cũ nếu cả 2 đều fail
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        img_result = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
        img_result[0:h1, 0:w1] = img1
        img_soKhop = cv2.drawMatches(img1, k1, img2, k2, matches, None, flags=2)
    else:
        # Sắp xếp theo score, chọn cao nhất
        results.sort(key=lambda x: x[0], reverse=True)
        best_score, img_result, best_H, best_mask, is_normal_order = results[0]
        
        # Vẽ ảnh so khớp với thứ tự đúng
        if is_normal_order:
            img_soKhop = cv2.drawMatches(img1, k1, img2, k2, matches, None, flags=2)
        else:
            # Đảo ngược matches để vẽ đúng (swap queryIdx và trainIdx)
            reversed_matches = []
            for m in matches:
                # Tạo match object mới với thứ tự đảo ngược
                reversed_match = cv2.DMatch()
                reversed_match.queryIdx = m.trainIdx
                reversed_match.trainIdx = m.queryIdx
                reversed_match.distance = m.distance
                reversed_matches.append(reversed_match)
            img_soKhop = cv2.drawMatches(img2, k2, img1, k1, reversed_matches, None, flags=2)

    # Cắt bỏ vùng đen bên phải
    gray = cv2.cvtColor(img_result, cv2.COLOR_RGB2GRAY)
    cols = np.where(gray.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        right = cols[-1]
        img_result = img_result[:, :right+1]
        # Cắt bỏ vùng đen bên trái nếu có
        left = cols[0] if len(cols) > 0 else 0
        if left > 0:
            img_result = img_result[:, left:]

    return img_soKhop, img_result

if __name__ == "__main__":
    # Mở các file ảnh
    img1 = cv2.imread('image5.jpg')
    img2 = cv2.imread('image6.jpg')

    h_min = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h_min / img1.shape[0]), h_min))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h_min / img2.shape[0]), h_min))

    #Convert sang chế độ màu RGB
    scr_imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    tar_imgRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_soKhop, img_Panorama = panorama(scr_imgRGB, tar_imgRGB)

    # Hiển thị 2 ảnh gốc
    fig=plt.figure(figsize=(16, 9))
    ax1,ax2 = fig.subplots(1, 2)

    ax1.imshow(scr_imgRGB)
    ax1.set_title('Ảnh 1')
    ax1.axis('off')

    ax2.imshow(tar_imgRGB)
    ax2.set_title('Ảnh 2')
    ax2.axis('off')
    plt.show()

    # Tạo cửa sổ hiển thị ảnh so khớp và ảnh Panorama
    fig=plt.figure(figsize=(16, 9))
    ax1,ax2 = fig.subplots(2, 1)
    ax1.imshow(img_soKhop)
    ax1.set_title('Ảnh so khớp ảnh 1 và ảnh 2')
    ax1.axis('off')

    ax2.imshow(img_Panorama)
    ax2.set_title('Ảnh Panorama tạo từ ảnh 1 và ảnh 2')
    ax2.axis('off')
    plt.show()
