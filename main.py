import argparse

import cv2
import sys
import numpy as np

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")

    return p.parse_args()


# 1. 전역 상수 및 템플릿 로드 함수
COIN_TEMPLATES = {} # 50/100/500원 앞/뒷면 템플릿을 저장할 딕셔너리

#def load_templates():
    # (TODO: 모든 템플릿 이미지 파일을 로드하고 COIN_TEMPLATES 딕셔너리에 저장)


# 동전 분류 함수 : 10원 여부
def is_ten_won(coin_roi):
    """
    (TODO: 10원인지 판별하여 True/False 반환)
    """
    return False

# 동전 분류 함수 : 50원, 100원, 500원
def classify_coin(coin_roi, templates):
    """
    (TODO: 템플릿 매칭을 통한 50원, 100원, 500원 분류)
    """
    return 0 # 분류된 금액 (50, 100, 500) 반환


"""
# 그레이스케일 변환
# 노이즈 제거
# 동전 검출
# (검출 된 모든 동전에 대해) 동전 분류
# 결과 계산 및 출력
"""
def main():
    args = parse_args()

    img = cv2.imread(args.input)
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    return 0

if __name__ == "__main__":
    sys.exit(main())
