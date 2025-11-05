import argparse

import cv2
import sys
import numpy as np

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")

    return p.parse_args()


def resize_with_aspect_ratio(image, width):
    """
    이미지 너비를 지정값으로 고정하고 비율을 유지하며 리사이징
    지정값보다 작은 이미지는 원본을 반환
    """
    (h, w) = image.shape[:2]

    # 너비가 1000보다 작거나 같으면 원본 반환
    if w <= width:
        return image

    # 비율 계산
    r = width / float(w)
    dim = (width, int(h * r))

    # 리사이징
    print(f"resizing : {w}x{h} -> {dim[0]}x{dim[1]}")
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def filter_nested_circles(circles):
    """
    자신보다 큰 원 안에 원점이 위치하는 원 모두 제거
    """
    if circles is None or len(circles) == 0:
        return []

    circles = circles[0, :]
    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)

    final_circles = []
    for current_circle in sorted_circles:
        x1, y1, r1 = current_circle
        is_nested = False

        for other_circle in final_circles:
            x2, y2, r2 = other_circle

            # 두 원의 중심 간의 거리
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # 1. current_circle(r1)이 other_circle(r2)보다 작아야 하고,
            # 2. current_circle의 중심이 other_circle의 반경 안에 있어야 함 (dist < r2)
            # 만족하면, 작은 원은 큰 원 내의 노이즈로 간주
            if r1 < r2 and dist < r2:
                is_nested = True
                break

        if not is_nested:
            final_circles.append(current_circle)

    return np.uint16(np.around(final_circles))

def main():
    args = parse_args()

    # 1. 이미지 로드
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: 이미지 로드 실패 {args.input}")
        return

    # 2. 리사이징
    img_resized = resize_with_aspect_ratio(img, width=800)
    # 최종 출력을 위한 원본 컬러 이미지 복사
    output_img = img_resized.copy()

    # 3. 그레이스케일 변환
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    clahe_img = clahe.apply(gray)

    # 5. 가우시안 블러링
    blurred = cv2.GaussianBlur(clahe_img, (13, 13), 0)

    # 6. 캐니 엣지 출력 (디버깅용)
    canny_debug = cv2.Canny(blurred, 60, 120)

    # 7. 허프 변환 원 검출
    # dp: 누적기 해상도 비율 (1 = 원본 크기, 1.2 or 1.5 = 해상도 줄여 속도 향상)
    # minDist: 검출된 원 중심 간의 최소 거리 (동전 반지름보다 약간 크게)
    # param1: Canny 엣지 검출기의 상위 임계값
    # param2: 누적기 임계값 (작을수록 많은 원, 클수록 정확한 원 검출)
    # minRadius: 검출할 원의 최소 반지름
    # maxRadius: 검출할 원의 최대 반지름
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=120,
        param2=45,
        minRadius=20,
        maxRadius=150
    )

    # 8. 원 안에 검출되는 원(노이즈) 제거
    if circles is not None:

        debug_output_img = img_resized.copy()
        raw_circles = np.uint16(np.around(circles))

        for i in raw_circles[0, :]:
            # 원 그리기 (노란색)
            cv2.circle(debug_output_img, (i[0], i[1]), i[2], (0, 255, 255), 2)
            # 중심점 (빨간색)
            cv2.circle(debug_output_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        count_text_raw = f"initial detection : {len(raw_circles[0])}"
        cv2.putText(debug_output_img, count_text_raw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        final_circles = filter_nested_circles(circles)

        print(f"초기 검출 원 개수: {len(circles[0])}, 최종 검출 원 개수: {len(final_circles)}")

        # 9. 화면에 검출된 원 표시
        for (x, y, r) in final_circles:
            # 원 그리기 (초록색)
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 3)
            # 중심점 (빨간색)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

        # 검출된 동전 개수 표시
        count_text = f"detection : {len(final_circles)}"
        cv2.putText(output_img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # 전처리 과정 출력
        # cv2.imshow("Grayscale", gray)
        # cv2.imshow("CLAHE", clahe_img)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Debug Canny Edges", canny_debug)
        cv2.imshow("DEBUG: Before Suppression", debug_output_img)
        cv2.imshow("Detected Coins", output_img)

    else:
        print("검출된 원 없음")
        cv2.imshow("No Detections", img_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
