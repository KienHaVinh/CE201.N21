import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract

# Khởi tạo đối tượng easyocr với các ngôn ngữ hỗ trợ là tiếng Việt và tiếng Anh
reader = easyocr.Reader(['vi', 'en'])

# Đặt đường dẫn tới tệp thực thi của Tesseract OCR
config = '--psm 7 --oem 3'
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Đọc file ảnh
file = "Image/test.png"
img = cv2.imread(file, 0)

# Tăng cường độ tương phản của ảnh sử dụng phương pháp CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_eq = clahe.apply(img)

# Ngưỡng hóa ảnh để chuyển đổi thành ảnh nhị phân
_, img_bin = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Invert ảnh nhị phân
img_bin = 255 - img_bin

# Lưu ảnh sau xử lý
cv2.imwrite("Image/cv_inverted.png", img_bin)

# Hiển thị ảnh sau xử lý
plotting = plt.imshow(img_bin, cmap='gray')
plt.show()

# Chiều dài của kernel là 1/100 chiều rộng tổng thể
kernel_len = np.array(img).shape[1] // 100

# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Sử dụng kernel dọc để phát hiện và lưu các đường dọc vào một tệp jpg
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite("Image/vertical.jpg", vertical_lines)

# Hiển thị ảnh đã tạo
plotting = plt.imshow(image_1, cmap='gray')
plt.show()

# Sử dụng kernel ngang để phát hiện và lưu các đường ngang vào một tệp jpg
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite("Image/horizontal.jpg", horizontal_lines)

# Hiển thị ảnh đã tạo
plotting = plt.imshow(image_2, cmap='gray')
plt.show()

# Kết hợp đường ngang và đường dọc thành một ảnh thứ ba, với cùng trọng số
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Erode và ngưỡng ảnh
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("Image/img_vh.jpg", img_vh)
bitxor = cv2.bitwise_xor(img, img_vh)
bitnot = cv2.bitwise_not(bitxor)
# Hiển thị ảnh đã tạo
plotting = plt.imshow(bitnot, cmap='gray')
plt.show()

# Phát hiện các đường viền để tìm hình chữ nhật
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def sort_contours(cnts, method="left-to-right"):
    # Khởi tạo cờ reverse và chỉ số sắp xếp
    reverse = False
    i = 0
    # Xử lý nếu chúng ta cần sắp xếp theo thứ tự ngược lại
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # Xử lý nếu chúng ta đang sắp xếp theo tọa độ y thay vì tọa độ x của hình chữ nhật
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # Xây dựng danh sách các hình chữ nhật đóng và sắp xếp chúng từ trên xuống
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # Trả về danh sách các đường viền và hình chữ nhật đã sắp xếp
    return (cnts, boundingBoxes)


# Sắp xếp tất cả các đường viền từ trên xuống
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
# Tạo một danh sách các chiều cao của tất cả các hình chữ nhật đã phát hiện
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# Lấy giá trị trung bình của chiều cao
mean = np.mean(heights)

# Tạo danh sách box để lưu tất cả các hình chữ nhật
box = []
# Lấy vị trí (x, y), chiều rộng và chiều cao cho mỗi đường viền và hiển thị đường viền trên ảnh
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w < 1000 and h < 500):
        image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box.append([x, y, w, h])
plotting = plt.imshow(image, cmap='gray')
plt.show()

# Tạo hai danh sách để xác định hàng và cột mà ô được đặt vào
row = []
column = []
j = 0
# Sắp xếp các hình chữ nhật vào hàng và cột tương ứng
for i in range(len(box)):
    if i == 0:
        column.append(box[i])
        previous = box[i]
    else:
        if box[i][1] <= previous[1] + mean / 2:
            column.append(box[i])
            previous = box[i]
            if i == len(box) - 1:
                row.append(column)
        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])

# Tính số lượng ô tối đa
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol
print(countcol)
# Lấy tâm của mỗi cột
center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
center = np.array(center)
center.sort()

# Dựa vào khoảng cách đến tâm cột, các hình chữ nhật được sắp xếp theo thứ tự tương ứng
finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)

# Từ mỗi ô hình ảnh đơn lẻ, các chuỗi được trích xuất bằng pytesseract và lưu trong một danh sách
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''

        # Kiểm tra xem ô có rỗng hay không
        if len(finalboxes[i][j]) == 0:
            outer.append(' ')
        else:
            # Lặp qua từng hình chữ nhật trong ô
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                    finalboxes[i][j][k][3]
                finalimg = bitnot[x:x + h, y:y + w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=2)
                out = pytesseract.image_to_string(erosion, config=config)

                # Kiểm tra xem kết quả có rỗng hay không
                if len(out) == 0:
                    out = pytesseract.image_to_string(erosion, config=config)
                inner = inner + ' ' + out
            outer.append(inner)

# Tìm tọa độ cạnh ngang thứ nhất
horizontal_sum = np.sum(image_2, axis=1)
first_column_index = np.argmax(horizontal_sum)

# Cắt ảnh phần thông tin thí sinh ra và lưu
processed_image_path = "Image/cv_inverted.png"
processed_image = cv2.imread(processed_image_path)
cropped_image = processed_image[0:first_column_index, :]
cropped_image_path = "Image/cropped_image.png"
cv2.imwrite(cropped_image_path, cropped_image)

# Đọc mã môn và mssv
text = reader.readtext('Image/cropped_image.png', detail=0)
ma_mon = text[2].split(":")[1].strip()
mssv = text[4].split(":")[1].strip()

# Tạo DataFrame từ danh sách chuỗi trích xuất và tạo chuỗi kết quả
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
dataframe.insert(0, ' ', range(len(dataframe)))
print(dataframe.to_string(index=False))
data = dataframe.style.set_properties(align="left")
result = ""
for i in range(len(dataframe)):
    if i == 0 or i == 2:
        continue
    for j in range(1, len(dataframe.columns)):
        value = dataframe.iloc[i, j]
        if value.strip() in ["A", "B", "C", "D"]:
            result += value.strip()
        else:
            result += "X"

# Xuất DataFrame ra tệp CSV
dataframe.to_csv("csv/text_extracted.csv", index=False, header=False)

# In thông tin và kết quả
print(ma_mon + "\n" + mssv + "\n" + result)
