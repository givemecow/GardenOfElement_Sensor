import cv2
import time
import json
import yaml

import pyk4a
import numpy as np
from helpers import colorize, colorize_grayscale, generate_mask
from pyk4a import Config, PyK4A
from osc_network import OscHandler

PLAY_SCALE = 500
BLACK_PIXEL_THRESHOLD = 0


def classify_box_type(width, height):
    if width == 0 or height == 0:
        return None

    ratio = max(width, height) / min(width, height)

    if ratio < 1.2:
        return 1
    elif ratio < 2.0:
        return 2
    else:
        return 3

def detect_and_draw_boxes(binary_img, display_img):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []

    if hierarchy is not None:
        for idx, h in enumerate(hierarchy[0]):
            if h[3] != -1:
                contour = contours[idx]
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                if min(w, h) >= 20 and max(w, h) >= 20:
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    box_cnt = box.reshape((-1, 1, 2))
                    similarity = cv2.matchShapes(contour, box_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                    if similarity < 0.15:
                        cx, cy = rect[0]
                        angle = rect[2]

                        if w < h :
                            tmp = w
                            w = h
                            h = tmp

                            angle += 90

                        box_type = classify_box_type(w, h)


                        detected_boxes.append((box_type, cx, cy, angle))


                        cv2.drawContours(display_img, [box], -1, (0, 255, 0), 2)
                        cv2.putText(display_img, f"{similarity:.3f}", (int(cx), int(cy)),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

                        info_text = f"x:{int(cx)}, y:{int(cy)}, angle:{angle:.1f}"
                        cv2.putText(display_img, info_text, (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return detected_boxes, display_img
def make_matrice(quad_points):
    src_pts = np.array(quad_points, dtype=np.float32)
    dst_pts = np.array([[0, 0], [PLAY_SCALE, 0], [PLAY_SCALE, PLAY_SCALE], [0, PLAY_SCALE]], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def calculate_black_pixel_ratio(binary_img):
    # 사람 유무 판단: 전체 중 검정 픽셀의 비율이 일정 이상이면 사람 있음
    black_pixels = cv2.countNonZero(cv2.bitwise_not(binary_img))  # 0 픽셀 개수
    black_ratio = black_pixels / (PLAY_SCALE ** 2)

    return black_ratio < BLACK_PIXEL_THRESHOLD

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():

    # 키넥트 세팅
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    # 변수 초기화
    no_person_frame_count = [0,0]  # 연속 비탐지 프레임 수
    matrices = [] # warping 매트릭스
    binary_image = [None, None]
    display_image = [None, None]
    generated_object = [[], []]

    # 캘리브레이션 데이터 조회
    with open("calibration_data.json", "r") as f:
        calibration_points = json.load(f)

    for i, quad in enumerate(calibration_points):
        matrices.append(make_matrice(quad))

    mask = cv2.imread("calibration_mask.png", cv2.IMREAD_GRAYSCALE)

    config = load_config()

    # OSC 네트워킹 세팅
    osc_handler = OscHandler(config)

    while True:
        capture = k4a.get_capture()

        if capture.transformed_ir is not None:
            img = generate_mask(capture.transformed_ir, threshold=2000)

            for i in range(2):
                binary_image[i] = cv2.warpPerspective(img, matrices[i], (PLAY_SCALE, PLAY_SCALE))
                display_image[i] = cv2.cvtColor(binary_image[i], cv2.COLOR_GRAY2BGR)

                person_present = calculate_black_pixel_ratio(binary_image[i])

                # print(f'player{i} frame {no_person_frame_count[i]}')

                # 연속 비탐지 프레임 카운트 갱신
                if person_present:
                    no_person_frame_count[i] = 0
                    cv2.putText(display_image[i], "Person Detected - Skipping Box", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                else:
                    no_person_frame_count[i] += 1

                    if no_person_frame_count[i] >= 90:

                        current_object, display_image[i] = detect_and_draw_boxes(binary_image[i], display_image[i])
                        remain_object = []

                        if generated_object[i] and current_object:
                            # print("generated_object and current_object is not null")
                            for gen_obj in generated_object[i]:
                                for cur_obj in current_object:
                                    if gen_obj[0] == cur_obj[0]:

                                        gen_type, gen_x, gen_y, gen_angle = gen_obj
                                        cur_type, cur_x, cur_y, cur_angle = cur_obj

                                        if (
                                                abs(gen_x - cur_x) < 10 and
                                                abs(gen_y - cur_y) < 10 and
                                                abs(gen_angle - cur_angle) < 10
                                        ):
                                            remain_object.append(gen_obj)
                                            remain_object.append(cur_obj)

                                            message = f'x: {abs(gen_x - cur_x)}, y: {abs(gen_y - cur_y)}, angle: {abs(gen_angle - cur_angle)}'

                            for gen_obj in generated_object[i]:
                                if gen_obj not in remain_object:
                                    message = f'remove player{i+1}/{gen_obj[0]}/{gen_obj[1]}/{gen_obj[2]}/{gen_obj[3]}'
                                    print(message)
                                    osc_handler.send_osc("/remove",message)
                                    generated_object[i].remove(gen_obj)

                            for cur_obj in current_object:
                                if cur_obj not in remain_object:
                                    message = f'add player{i}/{cur_obj[0]}/{cur_obj[1]}/{cur_obj[2]}/{cur_obj[3]}'
                                    print(message)
                                    osc_handler.send_osc("/add", message)
                                    generated_object[i].append(cur_obj)

                            # osc 통신으로 삭제할 오브젝트와 추가할 오브젝트 데이터 전송
                            # remove / player i / type / x / y / angle
                            # generated_object[i] 갱신

                        elif(generated_object[i]):
                            # print("current_object is null")
                            for gen_obj in generated_object[i]:
                                message = f'remove player{i}/{gen_obj[0]}/{gen_obj[1]}/{gen_obj[2]}/{gen_obj[3]}'
                                print(message)
                                osc_handler.send_osc("/remove", message)
                                generated_object[i].remove(gen_obj)
                            # 모든 오브젝트 삭제

                        elif(current_object):
                            # print("generated_object is null")
                            for cur_obj in current_object:
                                message = f'add player{i}/{cur_obj[0]}/{cur_obj[1]}/{cur_obj[2]}/{cur_obj[3]}'
                                print(message)
                                osc_handler.send_osc("/add", message)
                                generated_object[i].append(cur_obj)
                            # 모든 오브젝트 생성

                    else:
                        cv2.putText(display_image[i], "waiting_frame", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 디스플레이
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            merged = np.hstack((display_image[0], display_image[1]))

            # cv2.imshow("Transformed IR Masked", masked_img)
            # cv2.imshow("Box-Like Contours", img_children)
            cv2.imshow("display 0", display_image[0])
            cv2.imshow("display 1", display_image[1])
            cv2.imshow("generated", merged)

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break

    k4a.stop()


if __name__ == "__main__":
    main()
