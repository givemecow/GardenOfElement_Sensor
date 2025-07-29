import cv2
import json
import numpy as np
from kinect import Kinect

points = []

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

def save_points_to_json(points, filename="calibration_data.json"):
    grouped = [points[i:i + 4] for i in range(0, len(points), 4)]
    with open(filename, "w") as f:
        json.dump(grouped, f, indent=4)
    print(f"캘리브레이션 데이터 저장")

def make_mask(frame):
    global points

    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", mouse_callback)

    key = cv2.waitKey(1)

    mask = np.zeros_like(frame[:, :, 0])

    if len(points) / 4 >= 1:
        for i in range(int(len(points) / 4)):
            pts = np.array(points[i * 4:i * 4 + 4], dtype=np.int32)
            cv2.fillConvexPoly(mask, pts, 255)

    cv2.imshow("Mask", mask)

    if key == 13:  # Enter 키 -> 저장
        print("캘리브레이션 마스크 생성")
        cv2.destroyAllWindows()
        return mask

    return None

def test_calibration(frame, mask):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    mask_color = cv2.merge([mask, mask, mask]).astype(np.uint8)

    result = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
    cv2.imshow("Calibration Result", result)

    key = cv2.waitKey(1)

    if key == 13:  # Enter 키 -> 저장
        save_points_to_json(points)
        cv2.imwrite("calibration_mask.png", mask)
        # print("캘리브레이션 마스크 저장")
        return True

    return False

def calibrate():

    kinect = Kinect()
    kinect.device.start()

    mask = None

    while True:
        capture = kinect.device.get_capture()
        if capture.color is None:
            print("Error : capture.color is None")
            continue

        frame = capture.color.copy()

        if mask is None:
            mask = make_mask(frame)
        else:
            if(test_calibration(frame, mask)):
                break

    kinect.device.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
    print('종료')