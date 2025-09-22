#!/usr/bin/env python3
import cv2
import numpy as np
import time

def sample_video_frames(path, frame_interval=60, max_frames=2, resize_max_w=960):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    images, count, kept = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % frame_interval != 0:
            continue
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            continue
        h, w = frame.shape[:2]
        if resize_max_w and w > resize_max_w:
            new_w = resize_max_w
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))
        images.append(frame)
        kept += 1
        print(f"[collect] kept frame #{count} -> {frame.shape}")
        if max_frames and kept >= max_frames:
            break
    cap.release()
    return images

def make_stitcher(warper='cylindrical', exp_comp=True, conf_thresh=0.6):
    st = cv2.Stitcher.create()
    try:
        if hasattr(st, "setPanoConfidenceThresh"):
            st.setPanoConfidenceThresh(conf_thresh)
        if hasattr(st, "setWaveCorrection"):
            st.setWaveCorrection(True)
        # Warper
        if hasattr(cv2, "PyRotationWarper"):
            st.setWarper(cv2.PyRotationWarper(warper, 1.0))
        # Exposure compensator (can be slow)
        if hasattr(st, "setExposureCompensator"):
            comp = cv2.detail.ExposureCompensator_CHANNELS if exp_comp else cv2.detail.ExposureCompensator_NO
            st.setExposureCompensator(comp)
    except Exception as e:
        print(f"[warn] stitcher params: {e}")
    return st

def main():
    video = "Minecraft_stitch_test.mp4"
    print("[main] sampling...")
    images = sample_video_frames(video, frame_interval=60, max_frames=10, resize_max_w=960)
    if len(images) < 2:
        print("[main] not enough images to stitch")
        return

    print("[main] creating stitcher...")
    st = make_stitcher(warper='cylindrical', exp_comp=False, conf_thresh=0.6)

    print("[main] stitching...")
    t0 = time.time()
    status, pano = st.stitch(images)
    dt = time.time() - t0
    print(f"[main] status={status} time={dt:.2f}s  (0=OK,1=NEED_MORE,2=H_EST_FAIL,3=CAM_ADJ_FAIL)")

    if status == cv2.Stitcher_OK:
        cv2.imwrite("panorama.jpg", pano)
        print(f"[main] saved panorama.jpg size={pano.shape}")
        cv2.imshow("pano", pano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[hint] Try: more overlap (lower interval), fewer frames, smaller resize, or warper='spherical'.")

if __name__ == "__main__":
    main()
