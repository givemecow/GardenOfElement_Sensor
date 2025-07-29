from typing import Optional, Tuple

import cv2
import numpy as np

from pyk4a import ImageFormat

# 키넥트 컬러 포맷을 OpenCV용 BGRA 이미지로 변환
def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image

# 흑백이미지에 컬러맵을 적용해서 시각화
def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:

    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)

    return img

def colorize_grayscale(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
) -> np.ndarray:

    # 클리핑 범위 적용 (None이면 전체 사용)
    if clipping_range[0] is not None or clipping_range[1] is not None:
        img = image.clip(
            clipping_range[0] if clipping_range[0] is not None else image.min(),
            clipping_range[1] if clipping_range[1] is not None else image.max()
        )
    else:
        img = image.copy()

    # 0~255 범위로 정규화 (8비트 변환)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 이건 단일 채널 흑백 이미지임 (1채널)
    return img

def generate_mask(
        image: np.ndarray,
        threshold: int = 2000,
        invert: bool = True
) -> np.ndarray:

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 조건에 따라 마스크 생성
    if invert:
        mask = (image < threshold).astype(np.uint8) * 255
    else:
        mask = (image >= threshold).astype(np.uint8) * 255

    return mask