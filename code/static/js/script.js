// Preview ảnh khi chọn file
document.getElementById('image1').addEventListener('change', function(e) {
    previewImage(e.target.files[0], 'preview1');
});

document.getElementById('image2').addEventListener('change', function(e) {
    previewImage(e.target.files[0], 'preview2');
});

function previewImage(file, previewId) {
    if (file) {
        // Kiểm tra kích thước file (100MB)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            showError(`Ảnh ${previewId === 'preview1' ? '1' : '2'} quá lớn (${(file.size / 1024 / 1024).toFixed(2)}MB). Vui lòng chọn ảnh nhỏ hơn 100MB.`);
            // Reset input
            document.getElementById(previewId === 'preview1' ? 'image1' : 'image2').value = '';
            return;
        }
        
        const preview = document.getElementById(previewId);
        const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
        const fileName = file.name.toLowerCase();
        
        // Kiểm tra định dạng file RAW (DNG, CR2, NEF, ARW, etc.)
        const rawFormats = ['.dng', '.cr2', '.nef', '.arw', '.orf', '.raf', '.rw2', '.srw', '.pef', '.x3f'];
        const isRawFile = rawFormats.some(format => fileName.endsWith(format));
        
        // Nếu là file RAW hoặc file quá lớn (>20MB), bỏ qua preview
        if (isRawFile || file.size > 20 * 1024 * 1024) {
            const fileType = isRawFile ? 'RAW (DNG/CR2/NEF...)' : 'lớn';
            preview.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #667eea; border: 2px dashed #667eea; border-radius: 8px; background: #f8f9ff;">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">📷</div>
                    <div style="font-weight: 600; margin-bottom: 8px;">File đã chọn</div>
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                        <strong>${file.name}</strong>
                    </div>
                    <div style="font-size: 0.85em; color: #888; margin-bottom: 8px;">
                        Kích thước: ${fileSizeMB} MB
                        ${isRawFile ? '<br>Định dạng: RAW' : ''}
                    </div>
                    <div style="font-size: 0.75em; color: #28a745; margin-top: 10px; padding: 8px; background: #d4edda; border-radius: 5px;">
                        ✓ File hợp lệ - Vẫn có thể ghép ảnh bình thường
                    </div>
                    ${isRawFile ? '<div style="font-size: 0.7em; color: #999; margin-top: 5px;">(File RAW không thể preview trong trình duyệt)</div>' : ''}
                </div>
            `;
            preview.classList.add('show');
            return;
        }
        
        // Hiển thị loading cho file thông thường
        preview.innerHTML = '<div style="text-align: center; padding: 20px; color: #667eea;">Đang tải preview...</div>';
        preview.classList.add('show');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const originalDataUrl = e.target.result;
            const img = new Image();
            
            // Timeout để tránh treo nếu ảnh quá lớn
            const timeout = setTimeout(function() {
                // Nếu timeout, hiển thị ảnh gốc trực tiếp (fallback)
                preview.innerHTML = `
                    <img src="${originalDataUrl}" alt="Preview" style="max-width: 100%; height: auto; max-height: 400px; object-fit: contain;">
                    <div style="text-align: center; margin-top: 5px; font-size: 0.9em; color: #666;">
                        Kích thước: ${fileSizeMB} MB (Preview gốc)
                    </div>
                `;
            }, 5000); // 5 giây timeout
            
            img.onload = function() {
                clearTimeout(timeout);
                try {
                    // Tạo canvas để resize ảnh (giảm kích thước để hiển thị nhanh)
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // Giới hạn kích thước preview (max 800px chiều rộng)
                    const maxWidth = 800;
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > maxWidth) {
                        height = (height * maxWidth) / width;
                        width = maxWidth;
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    
                    // Vẽ ảnh đã resize lên canvas
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // Chuyển canvas thành data URL (JPG để nhẹ hơn)
                    const resizedDataUrl = canvas.toDataURL('image/jpeg', 0.85);
                    
                    // Hiển thị preview với thông tin kích thước file
                    preview.innerHTML = `
                        <img src="${resizedDataUrl}" alt="Preview" style="max-width: 100%; height: auto;">
                        <div style="text-align: center; margin-top: 5px; font-size: 0.9em; color: #666;">
                            Kích thước: ${fileSizeMB} MB
                        </div>
                    `;
                } catch (canvasError) {
                    // Nếu canvas fail, fallback về ảnh gốc
                    console.warn('Canvas resize failed, using original image:', canvasError);
                    preview.innerHTML = `
                        <img src="${originalDataUrl}" alt="Preview" style="max-width: 100%; height: auto; max-height: 400px; object-fit: contain;">
                        <div style="text-align: center; margin-top: 5px; font-size: 0.9em; color: #666;">
                            Kích thước: ${fileSizeMB} MB
                        </div>
                    `;
                }
            };
            
            img.onerror = function() {
                clearTimeout(timeout);
                // Nếu không load được ảnh, vẫn hiển thị thông tin file
                preview.innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #667eea; border: 2px dashed #667eea; border-radius: 8px; background: #f8f9ff;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">📷</div>
                        <div style="font-weight: 600; margin-bottom: 8px;">File đã chọn</div>
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">
                            <strong>${file.name}</strong>
                        </div>
                        <div style="font-size: 0.85em; color: #888; margin-bottom: 8px;">
                            Kích thước: ${fileSizeMB} MB
                        </div>
                        <div style="font-size: 0.75em; color: #28a745; margin-top: 10px; padding: 8px; background: #d4edda; border-radius: 5px;">
                            ✓ File hợp lệ - Vẫn có thể ghép ảnh bình thường
                        </div>
                        <div style="font-size: 0.7em; color: #999; margin-top: 5px;">
                            (Preview không khả dụng, nhưng vẫn có thể ghép ảnh)
                        </div>
                    </div>
                `;
            };
            
            img.src = originalDataUrl;
        };
        
        reader.onerror = function() {
            preview.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #c33;">
                    Lỗi đọc file. Vui lòng thử lại.
                </div>
            `;
        };
        
        reader.readAsDataURL(file);
    }
}

// Xử lý form submit
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const image1 = document.getElementById('image1').files[0];
    const image2 = document.getElementById('image2').files[0];
    
    if (!image1 || !image2) {
        showError('Vui lòng chọn đủ 2 ảnh');
        return;
    }
    
    formData.append('image1', image1);
    formData.append('image2', image2);
    
    // Hiển thị loading
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    
    submitBtn.disabled = true;
    btnText.textContent = 'Đang xử lý...';
    btnLoader.style.display = 'inline-block';
    
    // Ẩn kết quả cũ và lỗi
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        // Kiểm tra status code trước khi parse JSON
        if (response.status === 413) {
            showError('Ảnh quá lớn! Vui lòng chọn ảnh nhỏ hơn 100MB hoặc resize ảnh trước khi upload.');
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Hiển thị kết quả
            document.getElementById('resultImage1').src = data.image1;
            document.getElementById('resultImage2').src = data.image2;
            document.getElementById('matchImage').src = data.match_image;
            document.getElementById('panoramaImage').src = data.panorama;
            
            // Lưu ảnh panorama để download
            window.panoramaImageData = data.panorama;
            
            document.getElementById('results').style.display = 'block';
            
            // Scroll đến kết quả
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        } else {
            showError(data.error || 'Có lỗi xảy ra khi xử lý ảnh');
        }
    } catch (error) {
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        // Tắt loading
        submitBtn.disabled = false;
        btnText.textContent = 'Ghép Ảnh';
        btnLoader.style.display = 'none';
    }
});

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Download ảnh panorama
document.getElementById('downloadBtn').addEventListener('click', function() {
    if (window.panoramaImageData) {
        const link = document.createElement('a');
        link.href = window.panoramaImageData;
        link.download = 'panorama.jpg';
        link.click();
    }
});