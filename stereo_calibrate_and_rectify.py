import cv2
import numpy as np
import json
import time
import os
from pathlib import Path

# -------- CONFIG --------
CAMERA_LEFT_IDX  = 0    # /dev/video0
CAMERA_RIGHT_IDX = 1    # /dev/video1

CHECKERBOARD = (9, 6)     # inner corners
SQUARE_SIZE_MM = 20.0      # physical square size
REQUIRED_PAIRS = 12
DELAY_AFTER_CAPTURE = 0.5   # seconds
OUTPUT_JSON = "stereo_calib_result.json"
OUTPUT_NPZ  = "stereo_calib_maps.npz"
# ------------------------

# ---------- Helper ----------
def prepare_object_points(cb, square):
    cols, rows = cb
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return objp

def reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total = 0
    count = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total += err * err
        count += len(projected)
    return np.sqrt(total / count)

# ---------- MAIN ----------
def main():

    print("Checking available video devices:")
    os.system("ls /dev/video*")

    print("\nOpening cameras with V4L2 backend...")
    capL = cv2.VideoCapture(CAMERA_LEFT_IDX, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAMERA_RIGHT_IDX, cv2.CAP_V4L2)

    if not capL.isOpened():
        print("ERROR: Cannot open LEFT camera at index", CAMERA_LEFT_IDX)
        return
    if not capR.isOpened():
        print("ERROR: Cannot open RIGHT camera at index", CAMERA_RIGHT_IDX)
        return

    # Reduce resolution for stability (optional)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    objp = prepare_object_points(CHECKERBOARD, SQUARE_SIZE_MM)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    captured = 0
    last_cap_time = 0

    print("\n=== STEREO CALIBRATION MODE ===")
    print(f"Need {REQUIRED_PAIRS} captured pairs.")
    print("Press ESC to stop.\n")

    while captured < REQUIRED_PAIRS:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            print("Frame read failed. Is the USB power enough?")
            continue

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD)

        vis = np.hstack((frameL.copy(), frameR.copy()))

        if foundL:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(vis[:, :640], CHECKERBOARD, cornersL, foundL)

        if foundR:
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(vis[:, 640:], CHECKERBOARD, cornersR, foundR)

        cv2.putText(vis, f"Captured: {captured}/{REQUIRED_PAIRS}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Stereo Calibration (Left | Right)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        now = time.time()
        if foundL and foundR and (now - last_cap_time > DELAY_AFTER_CAPTURE):
            print(f"Captured pair {captured+1}")
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            objpoints.append(objp)
            captured += 1
            last_cap_time = now

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    if captured < 2:
        print("Not enough pairs to calibrate.")
        return

    print("\nCalibrating LEFT camera...")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpointsL, grayL.shape[::-1], None, None)

    print("Calibrating RIGHT camera...")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpointsR, grayR.shape[::-1], None, None)

    print("\nRunning stereoCalibrate...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-5)

    retS, CM1, DC1, CM2, DC2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        grayL.shape[::-1], flags=flags, criteria=criteria_stereo)

    print("Computing rectification maps...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        CM1, DC1, CM2, DC2, grayL.shape[::-1], R, T)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(CM1, DC1, R1, P1,
                                              grayL.shape[::-1], cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(CM2, DC2, R2, P2,
                                              grayR.shape[::-1], cv2.CV_16SC2)

    # compute mm-per-pixel (approx)
    last = imgpointsL[-1].reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
    h = []; v = []
    for r in range(CHECKERBOARD[1]):
        for c in range(CHECKERBOARD[0]-1):
            h.append(np.linalg.norm(last[r,c] - last[r,c+1]))
    for c in range(CHECKERBOARD[0]):
        for r in range(CHECKERBOARD[1]-1):
            v.append(np.linalg.norm(last[r,c] - last[r+1,c]))
    pix = (np.mean(h)+np.mean(v))/2
    mm_per_pixel = SQUARE_SIZE_MM / pix

    # Save JSON
    data = {
        "cameraMatrix1": CM1.tolist(),
        "distCoeffs1": DC1.tolist(),
        "cameraMatrix2": CM2.tolist(),
        "distCoeffs2": DC2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "mm_per_pixel": mm_per_pixel,
        "image_size": grayL.shape[::-1],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    np.savez(
        OUTPUT_NPZ,
        left_map1=mapL1, left_map2=mapL2,
        right_map1=mapR1, right_map2=mapR2,
        Q=Q, R=R, T=T
    )

    print("\n=== DONE ===")
    print("Saved:", OUTPUT_JSON)
    print("Saved:", OUTPUT_NPZ)
    print("mm_per_pixel =", mm_per_pixel)


if __name__ == "__main__":
    main()
