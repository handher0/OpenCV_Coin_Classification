import argparse
import cv2
import sys
import numpy as np
import os

# (parse_args, resize_to_width 함수는 동일)
def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")
    p.add_argument("--input", required=True, type=str, help="path to input image")
    return p.parse_args()

def resize_to_width(img, target_width):
    (h, w) = img.shape[:2]
    if w <= target_width:
        return img
    r = target_width / float(w)
    target_height = int(h * r)
    dim = (target_width, target_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def main():

    # --- [새 코드 1] ORB 분류기 및 템플릿 셋업 ---
    print("Initializing ORB detector and loading templates...")
    try:
        # ORB 검출기 생성 (특징점 1000개 제한)
        orb = cv2.ORB_create(nfeatures=1000)
        # BFMatcher 생성 (HAMMING 거리 사용, ORB에 적합)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        template_data = {}
        # 템플릿 파일 이름 목록 (사용자 제공 정보 반영)
        template_files = {
            "10_f": "10_won_2006_obverse.jpeg", "10_r": "10_won_2006_reverse.jpeg",
            "50_f": "50_won_1983_obverse.jpeg", "50_r": "50_won_1983_reverse.jpeg",
            "100_f": "100_won_1983_obverse.jpeg", "100_r": "100_won_1983_reverse.jpeg",
            "500_f": "500_won_1982_obverse.jpeg", "500_r": "500_won_1982_reverse.jpeg"
        }

        # 템플릿과 ROI를 이 크기로 표준화하여 크기 문제를 줄입니다.
        TEMPLATE_SIZE = (150, 150)

        for name, file in template_files.items():
            # 템플릿 로드 (그레이스케일)
            template_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                # 파일이 없을 경우 오류 출력
                print(f"Error: Template file not found at {file} (Current Dir: {os.getcwd()})", file=sys.stderr)
                # 템플릿 로드 실패 시 강제 종료
                # sys.exit(1) # 주석 처리하고 에러 파일만 스킵하도록 변경 가능
                continue

            template_img_resized = cv2.resize(template_img, TEMPLATE_SIZE)
            # 템플릿 특징점 및 기술자 계산
            kp, des = orb.detectAndCompute(template_img_resized, None)

            # 동전 값 추출 (예: '50_f' -> 50)
            value = int(name.split('_')[0])
            template_data[name] = {"kp": kp, "des": des, "value": value}

        if not template_data:
            print("Error: No templates were loaded successfully. Cannot proceed with ORB classification.", file=sys.stderr)
            return 1

        print(f"Templates loaded successfully: {len(template_data)} files.")

    except Exception as e:
        print(f"Error during classifier setup: {e}", file=sys.stderr)
        return 1
    # --- [셋업 끝] ---

    args = parse_args()
    img_original = cv2.imread(args.input)

    if img_original is None:
        print(f"Error: Image not found at {args.input}", file=sys.stderr)
        return 1

    # --- [검출을 위한 기본 설정] ---
    img = resize_to_width(img_original, 800)

    # 1. 색상 분류를 위한 HSV 이미지
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 2. 원 검출을 위한 그레이스케일 이미지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- [기존 검출 전처리] ---
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(14, 14))
    gray_clahe = clahe.apply(gray)

    kernel = np.ones((3,3), np.uint8)
    gray_eroded = cv2.erode(gray_clahe, kernel, iterations=1)

    blurred = cv2.GaussianBlur(gray_eroded, (13, 13), 0)
    canny_param1 = 120
    edges = cv2.Canny(blurred, canny_param1 // 2, canny_param1)

    # --- [기존 HoughCircles 검출] ---
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=canny_param1,
        param2=45,
        minRadius=15,
        maxRadius=150
    )

    counts = {500: 0, 100: 0, 50: 0, 10: 0}
    extraction_mask = np.zeros(gray.shape, dtype=np.uint8)

    if circles is not None:

        # --- [기존 NMS 로직] ---
        circles_list = circles[0]
        sorted_circles = circles_list[circles_list[:, 2].argsort()[::-1]]
        kept_circles = []

        for c_current in sorted_circles:
            x1, y1, r1 = c_current
            is_overlapping = False
            for c_kept in kept_circles:
                x_k, y_k, r_k = c_kept
                distance = np.sqrt((x1 - x_k)**2 + (y1 - y_k)**2)
                if distance < r_k:
                    is_overlapping = True
                    break
            if not is_overlapping:
                kept_circles.append(c_current)
        # --- [NMS 끝] ---

        print(f"Found {len(circles_list)} circles initially, filtered down to {len(kept_circles)}.")

        if kept_circles:
            final_circles_uint16 = np.uint16(np.around(kept_circles))

            # --- [HUE + SATURATION 기반 분류 로직] ---

            # [튜닝 값]
            HUE_MAX = 35        # 10원 동전 H값(31.05)을 포함하도록 35로 설정
            SATURATION_MIN = 35 # 최소 채도
            MIN_MATCH_THRESHOLD = 15 # [튜닝 필요] 매칭 성공으로 간주하는 최소 매칭 개수
            MATCH_DISTANCE_THRESHOLD = 75 # [튜닝 필요] 좋은 매치로 간주하는 최대 거리

            for (x, y, r) in final_circles_uint16:
                x, y, r = int(x), int(y), int(r)

                # 1. ROI 추출을 위한 원형 마스크 생성
                roi_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(roi_mask, (x, y), r, 255, -1)
                cv2.circle(extraction_mask, (x, y), r, 255, -1)

                # 원본 gray에서 ROI 추출 (특징점 매칭에 사용)
                # 경계 밖으로 나가지 않도록 np.clip을 사용하거나, r*2 크기의 새 이미지 생성
                y_start = max(0, y - r)
                y_end = min(gray.shape[0], y + r)
                x_start = max(0, x - r)
                x_end = min(gray.shape[1], x + r)
                roi_gray_cropped = gray[y_start:y_end, x_start:x_end]

                coin_label = "?"
                coin_value = 0

                # 2. 평균 H, S 값 계산
                mean_hsv_all = cv2.mean(hsv, mask=roi_mask)
                mean_hue = mean_hsv_all[0]
                mean_saturation = mean_hsv_all[1]

                print(f"DEBUG: Coin at ({x}, {y}) H={mean_hue:.2f}, S={mean_saturation:.2f}")

                is_10_won = (mean_saturation > SATURATION_MIN) and (mean_hue < HUE_MAX)

                # 3. 10원 조건: 채도(S)가 높고, 색상(H)이 갈색/노랑 계열일 때
                if is_10_won:
                    coin_value = 10
                    coin_label = "10"
                else:
                    coin_label = "Silver" # 다음 단계에서 50/100/500 분류 예정

                # --- 4. 결과 그리기 ---
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(img, coin_label, (x - r, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # --- [분류 로직 끝] ---

    else:
        print("No circles detected.")

    # --- (이하 동일) ---
    coins_only_gray = cv2.bitwise_and(gray, gray, mask=extraction_mask)
    coins_only_color = cv2.bitwise_and(img, img, mask=extraction_mask)

    total_amount = (counts[500] * 500) + (counts[100] * 100) + (counts[50] * 50) + (counts[10] * 10)
    print(f"500:{counts[500]}")
    print(f"100:{counts[100]}")
    print(f"50:{counts[50]}")
    print(f"10:{counts[10]}")
    print(f"Total Amount (10 Won only): {total_amount} Won")

    cv2.imshow("Detected Circles", img)
    cv2.imshow("Coin Mask (Binarized)", extraction_mask)
    cv2.imshow("Extracted Coins (Gray)", coins_only_gray)
    cv2.imshow("Extracted Coins (Color)", coins_only_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    sys.exit(main())