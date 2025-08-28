# -*- coding: utf-8 -*-
"""
Jetson Nano: MediaPipe 손 검출 → 로컬 마우스 커서/클릭
- 좌표: 손바닥 중심 EMA 스무딩 후 화면 좌표로 매핑
- 클릭: g/norm 라벨이 '0/1' 플래그를 포함(예: g:1?click, norm:0noclick)
        → 플래그를 파싱해서 (g_flag==1 or norm_flag==1) 일 때만 '엣지 + 디바운스'로 클릭 발사
- 포함:
  * 손목 원점화 + 손바닥 크기 스케일 정규화
  * TFLite 양자화 입출력 처리
  * 모델 출력 softmax(로그릿 대응)
  * HUD: g:, norm:, 플래그/판정 표시
"""

# sudo systemctl restart nvargus-daemon

from __future__ import print_function
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import re, json, argparse, collections, time
import numpy as np
import cv2
import mediapipe as mp
import pyautogui

# -------- 마우스 제어 기본 설정 --------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tflite", "keras"], default="tflite")
    ap.add_argument("--tflite", default="/home/han/artifacts/gesture_keras.tflite")
    ap.add_argument("--keras",  default="/home/han/artifacts/gesture_keras_best.keras")
    ap.add_argument("--norm",   default="/home/han/artifacts/norm.json")
    ap.add_argument("--labels", default="/home/han/artifacts/labels.txt")
    ap.add_argument("--histN", type=int, default=7, help="모드투표 윈도우(라벨 안정화)")
    ap.add_argument("--sensor", type=int, default=0)
    ap.add_argument("--width",  type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--send-fps", type=float, default=60.0)
    ap.add_argument("--exp-min-ns", type=int, default=50000)
    ap.add_argument("--exp-max-ns", type=int, default=20000000)
    ap.add_argument("--gain-min", type=float, default=1.0)
    ap.add_argument("--gain-max", type=float, default=16.0)
    ap.add_argument("--reopen-wait", type=float, default=0.25)
    ap.add_argument("--smooth-alpha", type=float, default=0.5,
                    help="0(무한평균)~1(스무딩 없음)")
    ap.add_argument("--invert-x", action="store_true", help="좌우 반전(미러 보정)")

    # 클릭: 플래그 기반 엣지 + 디바운스
    ap.add_argument("--debounce-sec", type=float, default=0.18,
                    help="엣지 감지 후 재클릭까지 최소 간격(sec)")

    # 포인터 범위 매핑
    ap.add_argument("--range-scale-x", type=float, default=1.6)
    ap.add_argument("--range-scale-y", type=float, default=1.6)
    ap.add_argument("--range-offset-x", type=float, default=0.0)
    ap.add_argument("--range-offset-y", type=float, default=0.0)
    return ap.parse_args()

def _safe_exp_max_ns(fps: int, user_max: int) -> int:
    period_ns = int(1e9 / max(fps, 1))
    safe = max(1000, min(user_max, period_ns - 1_000_000))
    return min(safe, 15_000_000)

def build_gst(sensor_id, w, h, fps, exp_min_ns, exp_max_ns, gain_min, gain_max):
    return (
        "nvarguscamerasrc sensor-id=%d "
        "exposuretimerange=\"%d %d\" gainrange=\"%.1f %.1f\" aelock=false "
        "! video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 "
        "! nvvidconv ! video/x-raw, format=BGRx "
        "! videoconvert ! video/x-raw, format=BGR "
        "! appsink drop=true max-buffers=1 sync=false"
    ) % (sensor_id, exp_min_ns, exp_max_ns, gain_min, gain_max, w, h, fps)

def open_cam(args):
    gst = build_gst(
        args.sensor, args.width, args.height, args.fps,
        args.exp_min_ns, _safe_exp_max_ns(args.fps, args.exp_max_ns),
        args.gain_min, args.gain_max
    )
    return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

# ---- 데이터/라벨 유틸 ----
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_norm(path):
    with open(path, "r", encoding="utf-8") as f:
        pj = json.load(f)
    return np.array(pj["mu"], np.float32), np.array(pj["sigma"], np.float32)

def _norm_label(s: str) -> str:
    s = (s or "").lower().strip()
    return "".join(ch for ch in s if ch.isalnum())

# ★★ 0/1 플래그라벨 파서: "1?click" / "0?no click" / "1click" / "0noclick" 모두 대응
def parse_flagged_label(raw: str):
    """
    returns: (flag, norm_text)
      - flag: 1/0/None (없으면 None)
      - norm_text: 영숫자 정규화된 본문 (예: "click", "noclick", "g0click" → "g0click")
    우선순위:
      1) 앞자리 '0/1' + '?' 패턴  ex) '1?click' -> flag=1, text='click'
      2) 정규화 후 맨 앞 '0/1'     ex) '0noclick' -> flag=0, text='noclick'
    """
    if raw is None:
        return None, ""
    s = raw.strip()
    # 패턴 1: ^([01])\?\s*(.*)
    m = re.match(r"^\s*([01])\?\s*(.*)$", s)
    if m:
        flag = int(m.group(1))
        text = m.group(2)
        return flag, _norm_label(text)

    # 패턴 2: 정규화 후 맨 앞 0/1
    n = _norm_label(s)
    m2 = re.match(r"^\s*([01])(.*)$", n)
    if m2:
        flag = int(m2.group(1))
        rest = _norm_label(m2.group(2))
        return flag, rest

    # 플래그 없음
    return None, _norm_label(s)

# ---- 전처리: 손목 원점화 + 손바닥 크기 스케일 정규화 ----
def preprocess_landmarks(vec63: np.ndarray) -> np.ndarray:
    lm = vec63.reshape(21, 3).astype(np.float32)
    wrist = lm[0].copy()
    lm[:, 0] -= wrist[0]; lm[:, 1] -= wrist[1]; lm[:, 2] -= wrist[2]
    palm_ids = [0, 1, 5, 9, 13, 17]
    palm = lm[palm_ids, :2]
    scale_vec = palm.max(axis=0) - palm.min(axis=0)
    palm_size = float(np.linalg.norm(scale_vec)) + 1e-6
    lm[:, :3] /= palm_size
    return lm.reshape(-1)

# ---- 소프트맥스(로그릿 대응) ----
def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32); x -= np.max(x); ex = np.exp(x)
    s = float(np.sum(ex)) or 1e-12
    return ex / s

def main():
    args = parse_args()
    mu, sg = load_norm(args.norm)
    labels = load_labels(args.labels)
    n_classes = len(labels)

    # ---- 모델 준비 ----
    if args.backend == "tflite":
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=args.tflite)
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter as _Interpreter
            interp = _Interpreter(model_path=args.tflite, experimental_delegates=[])
        interp.allocate_tensors()
        in_detail  = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        out_dim = int(out_detail["shape"][-1])
        assert out_dim == n_classes, f"라벨({n_classes}) != 출력차원({out_dim})"
        in_scale, in_zp = in_detail.get("quantization", (0.0, 0))
        out_scale, out_zp = out_detail.get("quantization", (0.0, 0))
        print("[INFO] TFLite input:", in_detail["dtype"], "quant:", (in_scale, in_zp))
        print("[INFO] TFLite output:", out_detail["dtype"], "quant:", (out_scale, out_zp))

        def infer(vec63_raw: np.ndarray) -> np.ndarray:
            vec63 = preprocess_landmarks(vec63_raw)
            x = ((vec63 - mu) / sg).reshape((1, 63)).astype(np.float32)
            if in_scale and in_scale > 0.0:
                x_q = np.round(x / in_scale + in_zp).astype(in_detail["dtype"])
            else:
                x_q = x.astype(in_detail["dtype"])
            interp.set_tensor(in_detail["index"], x_q)
            interp.invoke()
            y = interp.get_tensor(out_detail["index"])[0]
            if out_scale and out_scale > 0.0:
                y = (y.astype(np.float32) - out_zp) * out_scale
            return softmax(y)
    else:
        from tensorflow import keras
        model = keras.models.load_model(args.keras)
        out_dim = int(model.output_shape[-1])
        assert out_dim == n_classes, f"라벨({n_classes}) != 출력차원({out_dim})"
        def infer(vec63_raw: np.ndarray) -> np.ndarray:
            vec63 = preprocess_landmarks(vec63_raw)
            x = ((vec63 - mu) / sg).reshape((1, 63)).astype(np.float32)
            y = model.predict(x, verbose=0)[0].astype(np.float32)
            return softmax(y)

    print("[INFO] backend:", args.backend)
    print("[INFO] labels :", labels)
    print("[INFO] keys   : q or ESC to quit")

    # ---- MediaPipe ----
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = getattr(mp.solutions, "drawing_styles", None)
    landmark_style   = (mp_style.get_default_hand_landmarks_style()
                        if mp_style else mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
    connection_style = (mp_style.get_default_hand_connections_style()
                        if mp_style else mp_draw.DrawingSpec(color=(0,200,200), thickness=1))

    cap = open_cam(args)
    if not cap.isOpened():
        raise SystemExit("카메라 파이프라인 열기 실패 (nvargus-daemon 확인)")

    hist = collections.deque(maxlen=args.histN)
    do_flip = (not args.no_flip)

    alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
    has_ema = False
    ema_x = ema_y = 0.0

    # 클릭: 플래그 기반 엣지 + 디바운스
    last_is_click = False
    last_click_time = 0.0

    try:
        with mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7) as hands:

            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    time.sleep(args.reopen_wait)
                    cap = open_cam(args)
                    if not cap.isOpened():
                        print("카메라 재오픈 실패, 종료")
                        break
                    continue

                if do_flip:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                sx = sy = -1
                gesture_name = "-"
                gesture_norm = "-"

                g_flag = None   # ★ g:에서 뽑은 0/1
                n_flag = None   # ★ norm:에서 뽑은 0/1

                if res.multi_hand_landmarks:
                    hand = res.multi_hand_landmarks[0]

                    # ---- 분류 ----
                    raw_vec = []
                    for lm in hand.landmark:
                        raw_vec.extend([lm.x, lm.y, lm.z])
                    raw_vec = np.array(raw_vec, dtype=np.float32)
                    if raw_vec.size == 63:
                        prob = infer(raw_vec)
                        k = int(np.argmax(prob))
                        hist.append(k)
                        k_mode = max(set(hist), key=hist.count) if len(hist) > 0 else k
                        raw_label = labels[k_mode] if 0 <= k_mode < n_classes else "-"

                        # g: / norm: 을 이 포맷으로 만든다고 가정:
                        #   g:  -> "1?click" / "0?no click" 같은 원문 라벨 사용 가능
                        #   norm-> "1click" / "0noclick" 식으로 정규화 포맷 생성
                        # 실제 코드에선 labels.txt 원문을 g:로, 거기서 파싱한 걸 norm:로 표기
                        gesture_name = raw_label
                        # g 플래그/본문 파싱
                        g_flag, g_text_norm = parse_flagged_label(gesture_name)
                        # norm 표기는 "플래그 + 정규화본문"으로 만들자 (HUD용)
                        if g_flag is not None:
                            gesture_norm = f"{g_flag}{g_text_norm}"
                            n_flag = g_flag
                        else:
                            # g에 플래그가 없으면 본문에서 다시 시도
                            n_flag, n_text_norm = parse_flagged_label(gesture_name)
                            if n_flag is not None:
                                gesture_norm = f"{n_flag}{n_text_norm}"
                            else:
                                # 플래그가 정말 없으면 그냥 정규화만
                                gesture_norm = _norm_label(gesture_name)

                    # ---- 좌표: 손바닥 중심 ----
                    palm_ids = [0, 1, 5, 9, 13, 17]
                    cx = sum(hand.landmark[i].x for i in palm_ids) / len(palm_ids)
                    cy = sum(hand.landmark[i].y for i in palm_ids) / len(palm_ids)
                    x_px = int(cx * w); y_px = int(cy * h)

                    # ---- EMA ----
                    if x_px >= 0 and y_px >= 0:
                        if not has_ema:
                            ema_x, ema_y = float(x_px), float(y_px); has_ema = True
                        else:
                            ema_x = alpha * float(x_px) + (1.0 - alpha) * ema_x
                            ema_y = alpha * float(y_px) + (1.0 - alpha) * ema_y
                        sx, sy = int(round(ema_x)), int(round(ema_y))
                    else:
                        has_ema = False
                        sx, sy = -1, -1

                    # 디스플레이용 드로잉
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                           landmark_style, connection_style)
                    if sx >= 0 and sy >= 0:
                        cv2.circle(frame, (sx, sy), 6, (0, 255, 255), -1)

                # ---- 포인터 이동 ----
                if w > 1 and h > 1 and sx >= 0 and sy >= 0:
                    cx_img = (w - 1) * 0.5; cy_img = (h - 1) * 0.5
                    sx_scaled = cx_img + (sx - cx_img) * float(args.range_scale_x)
                    sy_scaled = cy_img + (sy - cy_img) * float(args.range_scale_y)
                    x_norm = (sx_scaled / (w - 1)) + float(args.range_offset_x)
                    y_norm = (sy_scaled / (h - 1)) + float(args.range_offset_y)
                    if args.invert_x: x_norm = 1.0 - x_norm
                    x_norm = float(np.clip(x_norm, 0.0, 1.0))
                    y_norm = float(np.clip(y_norm, 0.0, 1.0))
                    mx = int(round(x_norm * (SCREEN_W - 1)))
                    my = int(round(y_norm * (SCREEN_H - 1)))
                    pyautogui.moveTo(mx, my, duration=0)

                # ---- 클릭: 플래그(0/1) 기반 판정 (엣지 + 디바운스) ----
                # ★ 규칙: g_flag == 1 or n_flag == 1 → 클릭
                is_click = (g_flag == 0) or (n_flag == 0)
                now = time.time()
                if is_click and (not last_is_click) and (now - last_click_time) > args.debounce_sec:
                    try:
                        pyautogui.click()
                    except Exception as e:
                        print("[WARN] pyautogui.click() failed:", e)
                    last_click_time = now
                last_is_click = is_click

                # ---- HUD ----
                # g:에 원문(예: "1?click" / "0?no click")를 그대로, norm:엔 우리가 만든 "1click"/"0noclick"
                info = "px:{:4d},{:4d} | g:{:<14s} | norm:{:<14s} | gF:{} nF:{} | isClick:{} ".format(
                    sx if sx is not None else -1,
                    sy if sy is not None else -1,
                    (gesture_name or "")[:14],
                    (gesture_norm or "")[:14],
                    "-" if g_flag is None else g_flag,
                    "-" if n_flag is None else n_flag,
                    int(is_click)
                )
                cv2.putText(frame, info, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

                cv2.imshow("Hand → Local Mouse (Jetson)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass