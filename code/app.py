from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from Panorama2 import panorama
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size (tăng từ 16MB)

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def read_image(file):
    """Đọc ảnh từ file upload, hỗ trợ JPG, PNG, và các định dạng khác"""
    try:
        # Đọc bytes
        img_bytes = file.read()
        file.seek(0)  # Reset file pointer để có thể đọc lại nếu cần
        
        # Kiểm tra định dạng file
        filename = file.filename.lower() if hasattr(file, 'filename') else ''
        is_raw = any(filename.endswith(ext) for ext in ['.dng', '.cr2', '.nef', '.arw', '.orf', '.raf', '.rw2', '.srw', '.pef', '.x3f'])
        
        if is_raw:
            # Thử dùng PIL để đọc RAW (một số RAW được hỗ trợ)
            try:
                pil_image = Image.open(BytesIO(img_bytes))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
            except Exception as e:
                raise Exception(f"Không thể đọc file RAW. Vui lòng chuyển đổi sang JPG/PNG trước. Lỗi: {str(e)}")
        
        # Thử dùng PIL để đọc (hỗ trợ tốt PNG có alpha)
        try:
            pil_image = Image.open(BytesIO(img_bytes))
            # Chuyển RGBA sang RGB nếu có alpha channel
            if pil_image.mode == 'RGBA':
                # Tạo background trắng
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Dùng alpha làm mask
                pil_image = rgb_image
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Chuyển PIL Image sang numpy array
            img_array = np.array(pil_image)
            # PIL trả về RGB, OpenCV cần BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
        except:
            # Nếu PIL không đọc được, thử OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Thử đọc với flag khác
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    # Ảnh có alpha channel (BGRA), chuyển sang BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img is None:
                raise Exception("Không thể đọc ảnh. Vui lòng kiểm tra định dạng file (hỗ trợ: JPG, PNG, BMP, TIFF)")
            return img
    except Exception as e:
        raise Exception(f"Không thể đọc ảnh: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Vui lòng upload đủ 2 ảnh'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Vui lòng chọn ảnh'}), 400
        
        # Đọc ảnh với hàm hỗ trợ nhiều định dạng
        img1 = read_image(file1)
        img2 = read_image(file2)
        
        if img1 is None or img2 is None:
            return jsonify({'error': 'Không thể đọc ảnh. Vui lòng kiểm tra định dạng file (hỗ trợ: JPG, PNG, BMP, TIFF)'}), 400
        
        # Resize về cùng chiều cao
        h_min = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * h_min / img1.shape[0]), h_min))
        img2 = cv2.resize(img2, (int(img2.shape[1] * h_min / img2.shape[0]), h_min))
        
        # Chuyển BGR sang RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Gọi hàm panorama
        img_soKhop, img_panorama = panorama(img1_rgb, img2_rgb)
        
        # Chuyển ảnh kết quả về BGR để encode
        img_soKhop_bgr = cv2.cvtColor(img_soKhop, cv2.COLOR_RGB2BGR)
        img_panorama_bgr = cv2.cvtColor(img_panorama, cv2.COLOR_RGB2BGR)
        
        # Encode ảnh thành base64 (dùng PNG cho chất lượng tốt hơn, hoặc JPG để nhẹ hơn)
        # Có thể chọn .png hoặc .jpg tùy ý
        encode_format = '.jpg'  # hoặc '.png' nếu muốn chất lượng tốt hơn
        
        _, buffer_soKhop = cv2.imencode(encode_format, img_soKhop_bgr)
        _, buffer_panorama = cv2.imencode(encode_format, img_panorama_bgr)
        
        img_soKhop_base64 = base64.b64encode(buffer_soKhop).decode('utf-8')
        img_panorama_base64 = base64.b64encode(buffer_panorama).decode('utf-8')
        
        # Encode ảnh gốc để hiển thị
        _, buffer1 = cv2.imencode(encode_format, img1)
        _, buffer2 = cv2.imencode(encode_format, img2)
        img1_base64 = base64.b64encode(buffer1).decode('utf-8')
        img2_base64 = base64.b64encode(buffer2).decode('utf-8')
        
        # Xác định MIME type
        mime_type = 'image/jpeg' if encode_format == '.jpg' else 'image/png'
        
        return jsonify({
            'success': True,
            'image1': f'data:{mime_type};base64,' + img1_base64,
            'image2': f'data:{mime_type};base64,' + img2_base64,
            'match_image': f'data:{mime_type};base64,' + img_soKhop_base64,
            'panorama': f'data:{mime_type};base64,' + img_panorama_base64
        })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

# Xử lý lỗi 413 (Request Entity Too Large)
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Ảnh quá lớn! Vui lòng chọn ảnh nhỏ hơn 100MB hoặc resize ảnh trước khi upload.'}), 413

if __name__ == '__main__':
    app.run(debug=True, port=5000)