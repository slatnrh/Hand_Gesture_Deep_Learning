# collect_nano_keras.py  (Python 3.6 / Jetson Nano 호환)
# 손 랜드마크(21 x (x,y,z) = 63) + 정수 라벨을 CSV에 누적 저장하고,
# 종료 시 Keras용 X, y를 .npz로 저장

from __future__ import print_function
import os
import sys
import csv
import time
import argparse

import cv2
import mediapipe as mp
import numpy as np

# 기본 라벨 (숫자키 0~7로 선택)
DEFAULT_LABELS = [
    "no click",        # 0
    "click",           # 1
]

N_FEATS = 63  # 21 landmark * (x,y,z)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="gestures_keras.csv", help="출력 CSV 파일 경로")
    ap.add_argument("--npz", default="gestures_keras.npz", help="종료 시 저장할 NPZ 파일 경로(X,y)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--labels", default=None,
                    help="콤마로 구분한 커스텀 라벨 목록 (예: 'none,click,drag')")
    ap.add_argument("--flip", action="store_true",
                    help="좌우 반전 비활성(기본은 반전). 이 옵션 주면 반전하지 않음")
    ap.add_argument("--preview", action="store_true",
                    help="수집화면 미리보기(기본: 미리보기 ON). '--preview'를 주면 미리보기 OFF")
    return ap


def ensure_header(csv_path):
    """파일이 없으면 헤더를 씁니다. 있으면 건너뜁니다."""
    write_header = not os.path.exists(csv_path)
    f = open(csv_path, mode="a", newline="")  # Python 3.6에서 newline="" 권장
    w = csv.writer(f)
    if write_header:
        header = ["x{}".format(i) for i in range(N_FEATS)] + ["label_idx", "label_txt"]
        w.writerow(header)
    return f, w


def save_label_map(csv_path, labels):
    """정수라벨↔문자라벨 매핑을 텍스트로 저장 (재현성 확보)."""
    map_path = csv_path + ".labels.txt"
    with open(map_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(labels):
            f.write("{}\t{}\n".format(i, name))


def main():
    args = build_argparser().parse_args()

    # 라벨 목록
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    else:
        labels = list(DEFAULT_LABELS)

    # 현재 선택 라벨
    current_idx = 2 if "none" in labels else 0  # 기본 "none"이 있으면 그걸로 시작
    counts = dict((lbl, 0) for lbl in labels)
    recording = True

    # 파일 준비 (헤더 1회)
    f, w = ensure_header(args.csv)
    save_label_map(args.csv, labels)

    # 메모리 버퍼 (NPZ로 저장할 X, y)
    X_buf = []  # 각 원소 shape: (63,)
    y_buf = []  # 각 원소: int 라벨

    # MediaPipe 준비
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERR] 카메라({}) 열기 실패".format(args.cam))
        f.close()
        return 1

    print("[INFO] 저장 CSV:", os.path.abspath(args.csv))
    print("[INFO] 저장 NPZ:", os.path.abspath(args.npz))
    print("[INFO] 라벨:", ["{}:{}".format(i, l) for i, l in enumerate(labels)])
    print("[INFO] 조작법: 숫자키(0~{})로 라벨 전환, Space: 기록 토글, ESC: 종료".format(len(labels)-1))

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] 프레임 읽기 실패")
                    continue

                # 좌우반전(기본 ON) - 옵션으로 끌 수 있음
                if not args.flip:
                    frame = cv2.flip(frame, 1)

                h, w_img = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                # 랜드마크가 있을 때만 기록
                if res.multi_hand_landmarks and recording:
                    lm = res.multi_hand_landmarks[0].landmark
                    vec = []
                    for p in lm:
                        vec.extend([p.x, p.y, p.z])   # (0~1 범위, z는 상대깊이)
                    if len(vec) == N_FEATS:
                        label_txt = labels[current_idx]
                        # CSV: 피처 63 + 정수라벨 + 문자라벨
                        w.writerow(vec + [current_idx, label_txt])
                        counts[label_txt] = counts.get(label_txt, 0) + 1
                        # NPZ 버퍼에도 누적
                        X_buf.append(vec)
                        y_buf.append(current_idx)

                # 미리보기(기본 ON, --preview 주면 OFF)
                if not args.preview:
                    # 오버레이 표시 (라벨/REC/라벨별 카운트)
                    y0 = 25
                    cv2.putText(frame, "label: {} ({})".format(labels[current_idx], current_idx), (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y0 += 25
                    cv2.putText(frame, "REC: {}".format(int(recording)), (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y0 += 25
                    for i, lbl in enumerate(labels):
                        cv2.putText(frame, "{}:{}  [{}]".format(lbl, counts.get(lbl, 0), i),
                                    (10, y0 + 22*i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # 랜드마크 드로잉
                    if res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, res.multi_hand_landmarks[0],
                            mp_hands.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )

                    cv2.imshow("collect_nano (Keras)", frame)
                    k = cv2.waitKey(1) & 0xFF
                else:
                    # 미리보기 OFF일 때 키 입력만 폴링
                    k = cv2.waitKey(1) & 0xFF

                if k == 27:  # ESC
                    break

                # 숫자키로 라벨 전환
                if k != 255:  # 255는 입력 없음
                    for idx in range(len(labels)):
                        if k == ord(str(idx)):
                            current_idx = idx
                            print("→", labels[current_idx], "({})".format(current_idx))
                            break

                    # Space: 기록 토글
                    if k == ord(' '):
                        recording = not recording
                        print("REC:", recording)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        f.flush()
        f.close()

        # NPZ 저장
        if len(X_buf) > 0:
            X_arr = np.array(X_buf, dtype=np.float32)   # shape: (N, 63)
            y_arr = np.array(y_buf, dtype=np.int64)     # shape: (N,)
            try:
                np.savez_compressed(args.npz, X=X_arr, y=y_arr)
                print("[INFO] NPZ 저장 완료:", os.path.abspath(args.npz),
                      "  (X:", X_arr.shape, ", y:", y_arr.shape, ")")
            except Exception as e:
                print("[ERR] NPZ 저장 실패:", e)
        else:
            print("[WARN] 수집된 샘플이 없어 NPZ를 저장하지 않았습니다.")

        print("[INFO] 종료. CSV:", os.path.abspath(args.csv))

    return 0


if __name__ == "__main__":
    sys.exit(main())
