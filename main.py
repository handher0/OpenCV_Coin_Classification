import os
import re
import argparse
import cv2
import sys
import numpy as np

COIN_TEMPLATES = {
    10: [],
    50: [],
    100: [],
    500: []
}

orb = cv2.ORB_create(nfeatures=2000)

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


def load_templates(template_folder="templete", resize_dim=(200, 200)):
    """
    'templete' 폴더에서 템플릿 이미지를 로드 및
    ORB 디스크립터 추출하여 전역 COIN_TEMPLATES 딕셔너리에 저장
    """
    if not os.path.exists(template_folder):
        print(f"Error: Template folder not found at '{template_folder}'")
        return

    # 파일 이름 파싱을 위한 정규표현식
    pattern = re.compile(r"(\d+)([fb])\.(jpe?g|png)$", re.IGNORECASE)

    for filename in os.listdir(template_folder):
        match = pattern.match(filename)

        if not match:
            continue

        # 정규식 그룹에서 값 추출
        coin_value_str, side, extension = match.groups()
        coin_value = int(coin_value_str)

        if coin_value not in COIN_TEMPLATES:
            continue

        # 이미지 파일 경로
        filepath = os.path.join(template_folder, filename)

        # 템플릿 이미지를 그레이스케일로 로드
        template_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if template_img is None:
            print(f"Failed to load template: {filename}")
            continue

        try:
            template_resized = cv2.resize(template_img, resize_dim, interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            print(f"Failed to resize template {filename}: {e}")
            continue

        # 템플릿의 키포인트와 디스크립터 추출
        kp, des = orb.detectAndCompute(template_img, None)

        if des is not None:
            COIN_TEMPLATES[coin_value].append(des)
        else:
            print(f"Warning: No descriptors found for {filename}")
    print(f"  10 won: {len(COIN_TEMPLATES[10])} templates")
    print(f"  50 won: {len(COIN_TEMPLATES[50])} templates")
    print(f" 100 won: {len(COIN_TEMPLATES[100])} templates")
    print(f" 500 won: {len(COIN_TEMPLATES[500])} templates")


def is_ten(color_roi_cropped, x, y, r, ratio_threshold=0.6):
    """
    입력된 컬러 ROI가 10원짜리 동전(구리색)인지 색상으로 판별합니다.

    :param color_roi_cropped: bitwise_and로 추출된 원형의 BGR 컬러 ROI
    :param ratio_threshold: 10원으로 판단하기 위한 픽셀 비율 임계값
    :return: 10원이면 True, 아니면 False
    """
    # 1. ROI가 비어있는지 확인
    if color_roi_cropped is None or color_roi_cropped.size == 0:
        return False

    # 2. ROI를 BGR에서 HSV 컬러 스페이스로 변환
    hsv = cv2.cvtColor(color_roi_cropped, cv2.COLOR_BGR2HSV)

    # 3. 10원짜리 동전의 구리색/갈색 범위 정의 (HSV)
    #    H(색상): 5-25 (주황~갈색 계열)
    #    S(채도): 50-255 (너무 탁하지 않은 색)
    #    V(명도): 50-255 (너무 어둡거나 밝지 않은 색)
    #
    #    *** 중요: 이 값은 조명 환경에 따라 튜닝이 필요합니다! ***
    lower_copper = np.array([0, 90, 50])
    upper_copper = np.array([24, 255, 255])

    # 4. 지정된 색상 범위에 해당하는 픽셀 마스크 생성
    color_mask = cv2.inRange(hsv, lower_copper, upper_copper)

    # 5. ROI 영역(검은색 배경 제외)의 총 픽셀 수 계산
    #    color_roi_cropped는 이미 원형 마스킹이 적용되어 배경이 (0,0,0)입니다.
    #    그레이스케일로 변환 후, 0이 아닌 값(동전 영역)의 개수를 셉니다.
    gray_roi = cv2.cvtColor(color_roi_cropped, cv2.COLOR_BGR2GRAY)
    _ , coin_area_mask = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)
    total_coin_pixels = cv2.countNonZero(coin_area_mask)

    if total_coin_pixels == 0:
        return False  # 동전 영역이 없음

    # 6. 동전 영역 *내에서* 구리색 픽셀 수 계산
    #    (구리색 마스크) AND (동전 영역 마스크)
    copper_on_coin_mask = cv2.bitwise_and(color_mask, coin_area_mask)
    copper_pixel_count = cv2.countNonZero(copper_on_coin_mask)

    # 7. (디버깅용) 마스크 확인 - 필요하면 주석 해제
    # cv2.imshow(f"Check 10w - {np.random.randint(0, 100)}",
    #            np.hstack([color_roi_cropped,
    #                       cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR),
    #                       cv2.cvtColor(copper_on_coin_mask, cv2.COLOR_GRAY2BGR)]))

    # 8. 동전 영역 대비 구리색 픽셀의 비율 계산
    copper_ratio = copper_pixel_count / float(total_coin_pixels)

    # 9. 비율이 임계값(ratio_threshold)보다 높으면 10원으로 판단
    if copper_ratio > ratio_threshold:
        print(f"  [10원 판정] (x:{x}, y:{y}, r:{r} / 비율: {copper_ratio:.2f})")
        return True
    else:
        # (디버깅용) 10원이 아니라고 판단될 때 비율 출력
        # print(f"  [10원 아님] (비율: {copper_ratio:.2f})")
        return False


# BFMatcher(Brute-Force Matcher) 생성
# NORM_HAMMING: ORB, BRIEF 같은 바이너리 디스크립터에 사용
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def classify_coin(gray_roi_cropped, x, y, r, min_match_threshold=2, resize_dim=(200, 200)):
    """
    (개선된 버전)
    1. ROI 크기를 정규화 (Normalization)
    2. Lowe's Ratio Test를 이용한 knnMatch로 분류

    :param gray_roi_cropped: 검출된 동전의 그레이스케일 ROI
    :param min_match_threshold: 동전으로 판정하기 위한 최소 매칭 개수 (Ratio Test는 엄격하므로 4~5개로 낮춤)
    :param resize_dim: 정규화할 크기 (가로, 세로)
    :return: 분류된 동전 금액 (10, 50, 100, 500) 또는 0 (매칭 실패)
    """
    print(f"  [분류 시도] (x:{x}, y:{y}, r:{r})")

    # 1. (Solution 1) 크기 정규화 (Size Normalization)
    #    ROI가 너무 작거나(먼 거리) 너무 큰(가까운) 문제를 해결하기 위해
    #    모든 ROI를 일정한 150x150 크기로 리사이징합니다.
    try:
        roi_normalized = cv2.resize(gray_roi_cropped, resize_dim, interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"    -> [판정: 실패] (리사이징 오류: {e})")
        return 0

    # 2. 정규화된 이미지에서 ORB 특징점 추출
    kp_roi, des_roi = orb.detectAndCompute(roi_normalized, None)

    if des_roi is None or len(des_roi) < 2:  # knnMatch는 k=2 필요
        print(f"    -> [판정: 실패] (정규화된 ROI에서 특징점 검출 실패)")
        return 0

    best_match_score = 0
    best_match_value = 0

    # 3. 모든 템플릿과 비교
    for coin_value, template_descriptors_list in COIN_TEMPLATES.items():
        current_coin_best_score = 0

        for des_template in template_descriptors_list:

            # 4. (Solution 2) Lowe's Ratio Test (knnMatch)
            #    bf.match() 대신 bf.knnMatch(k=2)를 사용합니다.
            #    k=2: 각 ROI 특징점마다 가장 가까운 템플릿 특징점 2개를 찾습니다.
            try:
                matches = bf.knnMatch(des_template, des_roi, k=2)
            except cv2.error:
                continue  # 템플릿이나 ROI 디스크립터가 비어있을 때

            # 5. 좋은 매칭(good_matches) 선별
            good_matches = []
            ratio_threshold = 0.8  # (m.distance / n.distance) 비율 임계값

            for match_pair in matches:
                # k=2로 찾았지만, (가끔) 1개만 반환될 때가 있음
                if len(match_pair) < 2:
                    continue

                m, n = match_pair  # m: 1순위 매칭, n: 2순위 매칭

                # 1순위(m)가 2순위(n)보다 압도적으로 좋아야 '진짜' 매칭으로 인정
                # m.distance가 n.distance의 70%보다 작아야 함
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

            if len(good_matches) > current_coin_best_score:
                current_coin_best_score = len(good_matches)

        # (디버깅용) 각 동전별 최고 점수 출력
        print(f"    - {coin_value}원 점수: {current_coin_best_score}")

        if current_coin_best_score > best_match_score:
            best_match_score = current_coin_best_score
            best_match_value = coin_value

    # 6. 최종 판정 (Ratio Test는 점수가 훨씬 낮게 나옴!)
    #    '절대 개수'가 아닌 '엄선된 매칭'이므로, 임계값을 8 -> 4 정도로 낮춰야 함
    if best_match_score >= min_match_threshold:
        print(f"    -> [판정: {best_match_value}원] (Score: {best_match_score})")
        return best_match_value
    else:
        print(f"    -> [판정: 실패] (최고 점수: {best_match_score} < {min_match_threshold})")
        return 0

def main():
    args = parse_args()

    load_templates()

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
        total_amount = 0
        for (x, y, r) in final_circles:
            # 원 그리기 (초록색)
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 3)
            # 중심점 (빨간색)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

            # 원형 ROI 추출 로직 시작

            # 1. 원형 마스크 생성
            (h, w) = img_resized.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)  # 흰색으로 채워진 원

            # 2. 마스크를 이용해 ROI 추출
            #    'output_img' (컬러)에서 컬러 ROI 추출 (10원 분류용)
            color_roi = cv2.bitwise_and(output_img, output_img, mask=mask)

            #    'clahe_img' (그레이)에서 그레이스케일 ROI 추출 (50/100/500 분류용)
            gray_roi = cv2.bitwise_and(clahe_img, clahe_img, mask=mask)

            # 3. 마스킹된 영역만 잘라내기
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(w, x + r), min(h, y + r)

            color_roi_cropped = color_roi[y1:y2, x1:x2]
            gray_roi_cropped = gray_roi[y1:y2, x1:x2]

            # (디버깅용) 추출된 ROI 확인
            cv2.imshow(f"Color ROI {x}", color_roi_cropped)
            cv2.imshow(f"Gray ROI {x}", gray_roi_cropped)


            # --- [추가] 분류 로직 호출 (아직 함수 구현 안 됨) ---
            coin_value = 0
            #is_ten(gray_roi_cropped, x, y, r)
            coin_value = classify_coin(gray_roi_cropped, x, y, r)
            total_amount += coin_value

            # 분류 결과에 따라 원 색상 변경
            color = (0, 255, 0)  # 기본 (초록)
            if coin_value == 10:
                color = (0, 165, 255)  # 주황
            elif coin_value == 50:
                color = (255, 0, 0)  # 파랑
            elif coin_value == 100:
                color = (0, 0, 255)  # 빨강
            elif coin_value == 500:
                color = (255, 0, 255)  # 보라

            cv2.circle(output_img, (x, y), r, color, 3)

            text = str(coin_value) if coin_value > 0 else "?"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(output_img, text, (x - text_w // 2, y + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
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
