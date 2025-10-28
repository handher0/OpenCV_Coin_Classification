import argparse

import cv2
import sys

def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")

    # Sample argument usage
    p.add_argument("--input", required=True, type=str, help="path to input image")

    return p.parse_args()

def main():
    # Example code
    args = parse_args()
    img = cv2.imread(args.input)
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    # End of example code
    return 0

if __name__ == "__main__":
    sys.exit(main())
