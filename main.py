import sys
import json
import numpy as np
import cv2
import pyk4a
from pyk4a import PyK4A, Config
from rtmlib import BodyWithFeet
from halpe26 import halpe26
from utils import create_filters, get_pose, BoneConsistencyFilter, PullTestMonitor, RealTimeViewer
import time
from PyQt5.QtWidgets import QApplication
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

body_with_feet = BodyWithFeet(
    to_openpose=False, mode="performance", backend="onnxruntime", device="cuda"
)


def main():
    with open("properties.json", "r") as f:
        properties = json.load(f)

    # Initialize QApplication
    app = QApplication(sys.argv)

    clip_depth = [500.0, 5000.0]
    confidence_thr = 0.5
    radius = 2
    config = Config(
        color_format=pyk4a.ImageFormat.COLOR_BGRA32,
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
        synchronized_images_only=True,
        camera_fps=pyk4a.FPS.FPS_30,
        depth_delay_off_color_usec=0,
        wired_sync_mode=pyk4a.WiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=True,
    )
    k4a = PyK4A(config=config)
    k4a.start()

    frame_times = []  # List to store timestamps of the last frames

    # Initialize pose container and filters
    pose_frame = np.full((properties["pose"]["keypoints_count"], 5), np.nan)
    pose_sequence = np.copy(pose_frame)
    pose_filters = create_filters(
        num_keypoints=properties["pose"]["keypoints_count"],
        freq=pyk4a.FPS.FPS_30,
        min_cutoff=1.0,
        beta=0.3,
    )
    bone_filter = BoneConsistencyFilter(halpe26, fps=30)

    # Initialize RealTimeViewer and PullTestMonitor
    viewer = RealTimeViewer(halpe26)
    pull_test_monitor = PullTestMonitor(properties, halpe26)

    while True:
        capture = k4a.get_capture()
        current_time = time.time()
        frame_times.append(current_time)
        # Calculate the averaged FPS
        FPS = len(frame_times)
        # Remove frames older than 1 second
        frame_times = [t for t in frame_times if current_time - t <= 1.0]
        if np.any(capture.color):
            color = capture.color[:, :, :3]
            color = cv2.resize(color, (color.shape[1] // 4, color.shape[0] // 4))
            pointcloud = capture.transformed_depth_point_cloud
            pointcloud = cv2.resize(
                pointcloud, (pointcloud.shape[1] // 4, pointcloud.shape[0] // 4)
            )
            pointcloud = pointcloud.astype(float)
            mask = (pointcloud[:, :, 2] >= clip_depth[0]) & (pointcloud[:, :, 2] <= clip_depth[1])
            pointcloud[~mask] = np.nan

            keypoints, scores = body_with_feet(color)
            # Pick person with the highest overall score
            best_idx = np.nanargmax(np.nansum(scores, axis=1))
            keypoints, scores = keypoints[best_idx], scores[best_idx]
            pose_frame = get_pose(
                keypoints, scores, pointcloud, confidence_thr, radius, pose_frame, current_time
            )
            if keypoints is not None:
                # Apply filtering
                timestamp = time.time()
                filtered_pose = pose_frame

                valid_indices = ~np.isnan(pose_frame).any(axis=1)  # Check all dimensions at once
                for i in np.where(valid_indices)[0]:
                    filtered_pose[i][1] = pose_filters[i]["x"](pose_frame[i][1], timestamp)
                    filtered_pose[i][2] = pose_filters[i]["y"](pose_frame[i][2], timestamp)
                    filtered_pose[i][3] = pose_filters[i]["z"](pose_frame[i][3], timestamp)

                pose_frame = bone_filter(pose_frame)

                pose_sequence = np.dstack((pose_sequence, pose_frame))

                (
                    BOS,
                    xCOM,
                    stability,
                    baseline_met,
                    step_time,
                    nsteps,
                    step_events,
                    pull_time,
                    pull_magn,
                ) = pull_test_monitor.update(pose_sequence, FPS)

                # Update the viewer
                viewer.update(
                    pose_sequence[:, :, -1],
                    FPS,
                    bos=BOS,
                    xcom=xCOM,
                    stability=stability,
                    baseline_met=baseline_met,
                )

    k4a.stop()
    # Exit the QApplication
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
