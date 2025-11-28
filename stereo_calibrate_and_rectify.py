"""
stereo_calibrate_and_rectify.py

Usage:
- Place a checkerboard (default 9x6 inner corners, square size 20 mm) in view of both cameras.
- Position cameras rigidly on rig with known baseline (e.g., 200 mm). Ensure both cameras are synchronized as best as possible.
- Run: python stereo_calibrate_and_rectify.py
- The script will auto-capture stereo pairs when checkerboard is detected in both cameras.
- It collects a number of good pairs (default 12), then runs stereo calibration, rectification, and saves results to JSON + NumPy .npz with maps.

Features:
- Auto-capture when checkerboard found on both cameras.
- Computes intrinsic matrices, distortion coeffs, extrinsics (R,T), essential & fundamental matrices.
- Computes rectification maps and Q (reprojection to 3D) matrix.
- Estimates mm_per_pixel from detected corner spacing (left camera last view).
- Saves outputs to 'stereo_calib_result.json' and 'stereo_calib_maps.npz'.

Author: Generated for user (TA project)
"""

import cv2
import numpy as np
import json
import time
import os

# ---------- CONFIG ----------
CAMERA_LEFT_IDX = 0
CAMERA_RIGHT_IDX = 1
CHECKERBOARD = (9, 6)       # inner corners (cols, rows)
SQUARE_SIZE_MM = 20.0
REQUIRED_PAIRS = 12
DELAY_AFTER_CAPTURE = 0.4   # seconds between accepted captures
OUTPUT_JSON = "stereo_calib_result.json"
OUTPUT_NPZ = "stereo_calib_maps.npz"
# ----------------------------

def prepare_object_points(checkerboard, square_size):
    cols, rows = checkerboard
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

def reprojection_error_single(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_err = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_err += err**2
        total_points += len(imgpoints2)
    return np.sqrt(total_err / total_points) if total_points > 0 else 0.0

def main():
    print("Opening cameras:", CAMERA_LEFT_IDX, "and", CAMERA_RIGHT_IDX)
    capL = cv2.VideoCapture(CAMERA_LEFT_IDX)
    capR = cv2.VideoCapture(CAMERA_RIGHT_IDX)
    if not capL.isOpened() or not capR.isOpened():
        raise SystemExit("Cannot open one or both cameras. Check indices.")

    # Prepare storage
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    objp = prepare_object_points(CHECKERBOARD, SQUARE_SIZE_MM)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    captured = 0
    last_capture_time = 0

    print("Auto-capturing stereo pairs when checkerboard found simultaneously in both views.")
    print(f"Need {REQUIRED_PAIRS} good pairs. Press ESC to abort early.")

    while captured < REQUIRED_PAIRS:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Frame read failed from one of the cameras.")
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        visL = frameL.copy()
        visR = frameR.copy()
        if foundL:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, foundL)
        if foundR:
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, foundR)

        combined = np.hstack((visL, visR))
        cv2.putText(combined, f"Captured pairs: {captured}/{REQUIRED_PAIRS}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Stereo Capture (Left | Right)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        # Accept pair only if both views see checkerboard and not too fast after last capture
        if foundL and foundR and (time.time() - last_capture_time) > DELAY_AFTER_CAPTURE:
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            objpoints.append(objp)
            captured += 1
            last_capture_time = time.time()
            print(f"Captured pair {captured}/{REQUIRED_PAIRS}")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    if captured < 2:
        raise SystemExit("Not enough pairs captured to calibrate. Abort.")

    # Intrinsic calibrations for each camera first (single camera)
    print("Calibrating left camera...")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    print("Calibrating right camera...")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

    # Stereo calibration (find R, T between cameras)
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC  # we computed intrinsics; fix them
    print("Running stereoCalibrate (computing extrinsics R, T)...")
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        grayL.shape[::-1], criteria=criteria_stereo, flags=flags)

    print("Stereo calibrate done. Computing rectification...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                      cameraMatrix2, distCoeffs2,
                                                      grayL.shape[::-1], R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

    # Estimate mm_per_pixel using last captured left image's corners
    last_corners = imgpointsL[-1].reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
    h_dists = []
    v_dists = []
    for r in range(CHECKERBOARD[1]):
        for c in range(CHECKERBOARD[0]-1):
            p1 = last_corners[r,c]; p2 = last_corners[r,c+1]
            h_dists.append(np.linalg.norm(p2-p1))
    for c in range(CHECKERBOARD[0]):
        for r in range(CHECKERBOARD[1]-1):
            p1 = last_corners[r,c]; p2 = last_corners[r+1,c]
            v_dists.append(np.linalg.norm(p2-p1))
    mean_pix = (np.mean(h_dists) + np.mean(v_dists)) / 2.0
    mm_per_pixel = SQUARE_SIZE_MM / mean_pix

    # Reprojection error check
    errL = reprojection_error_single(objpoints, imgpointsL, rvecsL, tvecsL, cameraMatrix1, distCoeffs1)
    errR = reprojection_error_single(objpoints, imgpointsR, rvecsR, tvecsR, cameraMatrix2, distCoeffs2)

    # Save results
    out = {
        "camera_left_index": CAMERA_LEFT_IDX,
        "camera_right_index": CAMERA_RIGHT_IDX,
        "checkerboard": {"cols": CHECKERBOARD[0], "rows": CHECKERBOARD[1], "square_size_mm": SQUARE_SIZE_MM},
        "cameraMatrix1": cameraMatrix1.tolist(),
        "distCoeffs1": distCoeffs1.tolist(),
        "cameraMatrix2": cameraMatrix2.tolist(),
        "distCoeffs2": distCoeffs2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "mm_per_pixel": float(mm_per_pixel),
        "reproj_error_left_px": float(errL),
        "reproj_error_right_px": float(errR),
        "image_size": grayL.shape[::-1]
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Save rectification maps and matrices to npz
    np.savez(OUTPUT_NPZ,
             left_map1=left_map1, left_map2=left_map2,
             right_map1=right_map1, right_map2=right_map2,
             cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
             cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
             R=R, T=T, Q=Q, P1=P1, P2=P2)

    print("Stereo calibration results saved to:", OUTPUT_JSON)
    print("Rectification maps and matrices saved to:", OUTPUT_NPZ)
    print("Estimated mm_per_pixel (left cam):", mm_per_pixel)
    print("Reprojection errors (px): left=", errL, " right=", errR)
    print("Done. You can use 'stereo_rectify_and_reconstruct.py' to compute disparity and 3D point cloud.")

if __name__ == '__main__':
    main()
 