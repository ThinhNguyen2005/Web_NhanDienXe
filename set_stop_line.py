import cv2
import sys
import json

"""
Interactive stop-line selector.
Usage:
    python set_stop_line.py [image_path]

Controls:
 - Left click: select points (2 points define the stop line)
 - Move mouse after first click: live preview of the line
 - r: reset selection
 - s or Enter: save stop line (saves output.png and stop_line.json)
 - Esc or q: quit without saving

Output:
 - output.png: image with drawn stop line
 - stop_line.json: {"x1":..,"y1":..,"x2":..,"y2":..,"y_middle":..}

"""

img_path = sys.argv[1] if len(sys.argv) > 1 else 'anh1.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Không tìm thấy ảnh: {img_path}")
    sys.exit(1)

window_name = 'Đặt vạch dừng'
pts = []
mouse_pos = None


def draw_preview(base_img):
    out = base_img.copy()
    if pts:
        # draw first point
        cv2.circle(out, pts[0], 6, (0, 255, 0), -1)
    if len(pts) == 2:
        cv2.line(out, pts[0], pts[1], (0, 0, 255), 3)
        cv2.circle(out, pts[1], 6, (0, 255, 0), -1)
    elif len(pts) == 1 and mouse_pos is not None:
        # live preview line from first point to current mouse
        cv2.line(out, pts[0], mouse_pos, (0, 0, 255), 2)
    # show coordinates
    y = None
    if len(pts) == 2:
        y = int((pts[0][1] + pts[1][1]) / 2)
        cv2.putText(out, f'y_middle={y}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif len(pts) == 1:
        cv2.putText(out, 'Click second point or press r to reset', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(out, 'Click two points to set stop line', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return out


def mouse_cb(event, x, y, flags, param):
    global pts, mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 2:
            pts.append((x, y))
            print('Point', len(pts), 'set to', (x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_cb)

print('Click chuột trái để chọn 2 điểm làm vạch dừng.')
print('Nhấn r để reset, s/Enter để lưu, Esc/q để thoát không lưu.')

while True:
    preview = draw_preview(img)
    cv2.imshow(window_name, preview)
    key = cv2.waitKey(20) & 0xFF
    if key == 27 or key == ord('q'):
        print('Thoát không lưu.')
        break
    elif key == ord('r'):
        pts = []
        mouse_pos = None
        print('Reset selection.')
    elif key == ord('s') or key == 13:
        if len(pts) == 2:
            # draw final line and save
            out = img.copy()
            cv2.line(out, pts[0], pts[1], (0, 0, 255), 3)
            cv2.circle(out, pts[0], 6, (0, 255, 0), -1)
            cv2.circle(out, pts[1], 6, (0, 255, 0), -1)
            y_middle = int((pts[0][1] + pts[1][1]) / 2)
            cv2.putText(out, f'y_middle={y_middle}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite('output.png', out)
            data = {
                'x1': pts[0][0], 'y1': pts[0][1],
                'x2': pts[1][0], 'y2': pts[1][1],
                'y_middle': y_middle,
                'image': img_path
            }
            with open('stop_line.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print('Đã lưu output.png và stop_line.json')
            break
        else:
            print('Bạn chưa chọn đủ 2 điểm. Nhấn r để reset hoặc tiếp tục chọn.')

cv2.destroyAllWindows()
