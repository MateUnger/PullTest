import numpy as np
from OneEuroFilter import OneEuroFilter
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QFrame,
)
from PyQt5.QtCore import Qt
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from pyqtgraph import PlotWidget, ScatterPlotItem, PlotDataItem
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from collections import deque
from scipy.signal import savgol_filter, find_peaks

import importlib

rtmlib_module = importlib.import_module("rtmlib")


def create_filters(
    num_keypoints: int, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.0
) -> list[OneEuroFilter]:
    """
    Create a list of OneEuroFilters for each dimension of a keypoint (x,y,z).

    Args:
        num_keypoints: number of keypoints
        freq: frequency of the time series
        min_cutoff: min cutoff frequency in Hz
        beta: parameter to reduce latency

    Returns:
        filters: list of OneEuroFilters (num_keypoints,3)
    """
    # Create filters for smoothing keypoints
    filters = []
    for _ in range(num_keypoints):
        filters.append(
            {
                "x": OneEuroFilter(freq, min_cutoff, beta),
                "y": OneEuroFilter(freq, min_cutoff, beta),
                "z": OneEuroFilter(freq, min_cutoff, beta),
            }
        )
    return filters


def get_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    pointcloud: np.ndarray,
    kpt_thr: float,
    radius: int,
    pose_frame: np.ndarray,
    timestamp: float,
):
    """
    Searches for 3D points within pointcloud (from Kinect) corresponding to 2D keypoints. Returns keypoints converted to 3D.

    Args:
        keypoints: keypoint data from pose estimation model (num_kpts,2)
        scores: confidence scores associated with keypoints (num_kpts, 1)
        pointscloud: 3D data for a frame from Kinect
        kpt_thr: confidence threshold for keypoint confidences
        radius: search radius in pointcloud pixels
        pose_frame: ??????
        timestamp: timestamp of the generated data

    Returns:
        keypoints_3d: 3D keypoints with timestamps and confidence scores.
    """
    # Extract 3D pose information from keypoints and point cloud
    keypoints_3d = np.full_like(pose_frame, np.nan)
    num_kpts = keypoints.shape[0]

    if np.any(keypoints):

        for kpt_idx in range(num_kpts):

            if scores[kpt_idx] > kpt_thr:
                kpt_pixel = keypoints[kpt_idx]

                # if source pixel falls within pointcloud boundaries
                if kpt_pixel[0] < pointcloud.shape[1] and kpt_pixel[1] < pointcloud.shape[0]:
                    for j in range(radius + 1):
                        y_min = max(0, int(kpt_pixel[1]) - j)
                        y_max = min(pointcloud.shape[0], int(kpt_pixel[1]) + j + 1)
                        x_min = max(0, int(kpt_pixel[0]) - j)
                        x_max = min(pointcloud.shape[1], int(kpt_pixel[0]) + j + 1)

                        points = pointcloud[y_min:y_max, x_min:x_max, :]

                        # there is at least 1 valid point in pointcloud
                        if not np.isnan(points).all():
                            average_point = np.nanmean(points, axis=(0, 1))

                            # TODO: check if this is the correct logic
                            # there is at least 1 valid coordinate, stop searching
                            if not np.isnan(average_point).all():
                                break
                        else:
                            average_point = np.full(3, np.nan)

                    # reorder dims, flip depth axis, convert to meters
                    coordinates = average_point[[0, 2, 1]]
                    coordinates[2] = coordinates[2] * -1
                    coordinates = coordinates / 1000

                    # add timestamp, 3d kpt, confidence score to output
                    coordinates = np.hstack(([timestamp], coordinates, [scores[kpt_idx]]))
                    keypoints_3d[kpt_idx, :] = coordinates
    return keypoints_3d


class BoneConsistencyFilter:
    def __init__(
        self,
        halpe26: dict,
        fps: float = 30,
        buffer_duration: float = 1.0,
        min_valid_duration: float = 0.3,
        max_deviation: float = 0.3,
    ):
        """
        Args:
            halpe26 (dict): skeleton definition
            fps (float): frames per second
            buffer_duration (float): how many seconds of bone length data to keep in buffer
            min_valid_duration (float): how many seconds of samples are needed to trust a bone baseline
            max_deviation (float): allowed deviation from baseline in % (e.g., 0.3 for ±30%)
        """
        self.fps = fps
        self.buffer_size = int(buffer_duration * fps)
        self.min_valid_samples = int(min_valid_duration * fps)
        self.max_deviation = max_deviation

        # Build bone list from halpe26
        name_to_id = {v["name"]: k for k, v in halpe26["keypoint_info"].items()}
        self.bones = [
            (name_to_id[link[0]], name_to_id[link[1]])
            for link in [v["link"] for v in halpe26["skeleton_info"].values()]
            if link[0] in name_to_id and link[1] in name_to_id
        ]

        self.bone_length_buffers = {bone: deque(maxlen=self.buffer_size) for bone in self.bones}
        self.bone_baselines = {}

    def update(self, pose_frame):
        for bone in self.bones:
            i, j = bone
            pi, pj = pose_frame[i][1:4], pose_frame[j][1:4]
            if not np.any(np.isnan(pi)) and not np.any(np.isnan(pj)):
                length = np.linalg.norm(pi - pj)
                self.bone_length_buffers[bone].append(length)
                if len(self.bone_length_buffers[bone]) >= self.min_valid_samples:
                    self.bone_baselines[bone] = np.median(self.bone_length_buffers[bone])

    def filter(self, pose_frame):
        for bone in self.bones:
            if bone not in self.bone_baselines:
                continue
            i, j = bone
            pi, pj = pose_frame[i][1:4], pose_frame[j][1:4]
            if not np.any(np.isnan(pi)) and not np.any(np.isnan(pj)):
                current_length = np.linalg.norm(pi - pj)
                baseline = self.bone_baselines[bone]
                deviation = abs(current_length - baseline) / baseline
                if deviation > self.max_deviation:
                    pose_frame[i, 1:4] = np.nan
                    pose_frame[j, 1:4] = np.nan
        return pose_frame

    def __call__(self, pose_frame):
        self.update(pose_frame)
        return self.filter(pose_frame)


class PullTestMonitor:
    def __init__(self, properties, halpe26):
        """
        Initialize the PullTestMonitor.

        Args:
            properties (dict): Configuration properties from properties.json.
            halpe26 (dict): Skeleton definition.
        """
        self.properties = properties
        self.halpe26 = halpe26
        self.baseline_thr = properties["pull_test"]["baseline_thr"]
        self.baseline_win = properties["pull_test"]["baseline_win"]
        self.shoulder_kpts = [
            self.get_keypoint_id(properties["pull_test"]["shoulder_kpts"][0]),
            self.get_keypoint_id(properties["pull_test"]["shoulder_kpts"][1]),
        ]
        self.pull_thr = properties["pull_test"]["pull_thr"]
        self.ankle_kpts = [
            self.get_keypoint_id(properties["step_detection"]["ankle_kpts"][0]),
            self.get_keypoint_id(properties["step_detection"]["ankle_kpts"][1]),
        ]
        self.step_height_thr = properties["step_detection"]["step_height_thr"]
        self.step_vel_thr = properties["step_detection"]["step_vel_thr"]
        self.step_len_thr = properties["step_detection"]["step_len_thr"]

    def get_keypoint_id(self, keypoint_name):
        """
        Get the keypoint ID from the halpe26 skeleton definition.

        Args:
            keypoint_name (str): Name of the keypoint.

        Returns:
            int: Keypoint ID.
        """
        for kpt_id, kpt_info in self.halpe26["keypoint_info"].items():
            if kpt_info["name"] == keypoint_name:
                return kpt_id
        raise ValueError(f"Keypoint '{keypoint_name}' not found in halpe26.")

    def smooth_keypoints(self, pose_sequence, window_length=5, polyorder=2):
        """
        Smooth 3D keypoint sequences using a Savitzky-Golay filter.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints) with shape (num_keypoints, 4, num_frames).
            window_length (int): The length of the filter window (must be odd and >= polyorder + 1).
            polyorder (int): The order of the polynomial used to fit the samples.

        Returns:
            np.ndarray: Smoothed pose sequence with the same shape as the input.
        """
        # Ensure pose_sequence has 4 dimensions (timestamp, x, y, z)
        if pose_sequence.shape[1] < 4:
            padding = np.full(
                (pose_sequence.shape[0], 4 - pose_sequence.shape[1], pose_sequence.shape[2]), np.nan
            )
            pose_sequence = np.concatenate((pose_sequence, padding), axis=1)

        smoothed_sequence = np.copy(pose_sequence)

        for kpt_idx in range(pose_sequence.shape[0]):
            for dim in range(1, 4):  # Smooth x, y, z dimensions (skip timestamp)
                if not np.isnan(pose_sequence[kpt_idx, dim, :]).all():
                    smoothed_sequence[kpt_idx, dim, :] = savgol_filter(
                        pose_sequence[kpt_idx, dim, :],
                        window_length=window_length,
                        polyorder=polyorder,
                        mode="nearest",
                    )

        return smoothed_sequence

    def ensure_baseline(self, pose_sequence, fps):
        """
        Ensure the baseline condition by checking the 3D acceleration of the shoulders.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.

        Returns:
            bool: True if acceleration is below the threshold for the baseline window, False otherwise.
        """
        pose_sequence = self.smooth_keypoints(pose_sequence)
        required_frames = int(fps * self.baseline_win)
        if pose_sequence.shape[2] < required_frames:
            return False  # Not enough data available

        # Extract shoulder positions over the baseline window
        right_shoulder = pose_sequence[self.shoulder_kpts[0], 1:4, -required_frames:]
        left_shoulder = pose_sequence[self.shoulder_kpts[1], 1:4, -required_frames:]

        # Check if there are enough valid points for gradient calculation
        if right_shoulder.shape[1] < 2 or left_shoulder.shape[1] < 2:
            return False  # Not enough data points for gradient calculation

        # Compute 3D velocities
        right_velocity = np.gradient(right_shoulder, axis=1) * fps
        left_velocity = np.gradient(left_shoulder, axis=1) * fps

        # Compute 3D accelerations
        right_acceleration = np.gradient(right_velocity, axis=1) * fps
        left_acceleration = np.gradient(left_velocity, axis=1) * fps

        # Compute magnitudes of accelerations
        right_acc_magnitude = np.linalg.norm(right_acceleration, axis=0)
        left_acc_magnitude = np.linalg.norm(left_acceleration, axis=0)

        # Check if both accelerations are below the threshold
        if np.all(right_acc_magnitude < self.baseline_thr) and np.all(
            left_acc_magnitude < self.baseline_thr
        ):
            return True
        return False

    def compute_BOS(self, pose_sequence, fps):
        """
        Computes the Base of Support (BOS) based on keypoints and properties.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.

        Returns:
            np.ndarray: Computed BOS or NaN-filled array if data is invalid.
        """
        pose_sequence = self.smooth_keypoints(pose_sequence)
        keypoints_left = ["left_" + kpt for kpt in self.properties["BOS"]["keypoints"]]
        keypoints_right = ["right_" + kpt for kpt in self.properties["BOS"]["keypoints"]]
        keypoints_kpts = [
            kpt_id
            for kpt in keypoints_left + keypoints_right
            for kpt_id, kpt_info in self.halpe26["keypoint_info"].items()
            if kpt_info["name"] == kpt
        ]

        pose_sequence = np.copy(pose_sequence)
        frames = min([int(fps * self.properties["stream"]["average_win"]), pose_sequence.shape[2]])
        pose_sequence = pose_sequence[keypoints_kpts, :, -frames:]

        bos = pose_sequence[:, 1:3, :]  # Extract x and y coordinates
        if bos.size == 0 or np.isnan(bos).all():
            return np.full((8, 2), np.nan)  # Return NaN-filled BOS if data is invalid

        # Compute height differences
        height_left = np.nanmean(pose_sequence[: len(keypoints_left), 3, :], axis=1)
        height_right = np.nanmean(pose_sequence[len(keypoints_left) :, 3, :], axis=1)
        height_diff = height_left - height_right

        # Check height differences and adjust BOS
        for i in range(len(height_diff)):
            if np.abs(height_diff[i]) > self.properties["step_detection"]["step_height_thr"]:
                if height_diff[i] > 0:
                    bos[i, :, :] = np.nan
                else:
                    bos[i + len(keypoints_left), :, :] = np.nan

        # Optionally average BOS over frames
        bos = (
            np.nanmean(bos, axis=2)
            if bos.size > 0 and not np.isnan(bos).all() and np.any(~np.isnan(bos))
            else np.full((8, 2), np.nan)
        )
        return bos

    def compute_xCOM(self, pose_sequence, fps):
        """
        Computes the extrapolated Center of Mass (xCOM) based on pose sequence and properties.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.

        Returns:
            np.ndarray: Computed xCOM or NaN-filled array if data is invalid.
        """
        pose_sequence = self.smooth_keypoints(pose_sequence)
        pose_sequence = np.copy(pose_sequence)
        frames = min([int(fps * self.properties["stream"]["average_win"]), pose_sequence.shape[2]])
        pose_sequence = pose_sequence[:, :, -frames:]

        # Leg length
        left_leg = ["left_" + s for s in self.properties["COM"]["leg_length"]]
        right_leg = ["right_" + s for s in self.properties["COM"]["leg_length"]]
        left_leg_kpts = [
            kpt_id
            for kpt in left_leg
            for kpt_id, kpt_info in self.halpe26["keypoint_info"].items()
            if kpt_info["name"] == kpt
        ]
        right_leg_kpts = [
            kpt_id
            for kpt in right_leg
            for kpt_id, kpt_info in self.halpe26["keypoint_info"].items()
            if kpt_info["name"] == kpt
        ]

        if len(left_leg_kpts) < 2 or len(right_leg_kpts) < 2:
            return np.full((2,), np.nan)  # Return NaN-filled xCOM if data is invalid

        left_leg_length = np.nanmean(
            np.linalg.norm(
                pose_sequence[left_leg_kpts[0], 1:4, :] - pose_sequence[left_leg_kpts[1], 1:4, :],
                axis=0,
            )
        )
        right_leg_length = np.nanmean(
            np.linalg.norm(
                pose_sequence[right_leg_kpts[0], 1:4, :] - pose_sequence[right_leg_kpts[1], 1:4, :],
                axis=0,
            )
        )

        if np.isnan(left_leg_length) or np.isnan(right_leg_length):
            return np.full((2,), np.nan)

        l = np.asarray(
            [
                np.nanmean([left_leg_length, right_leg_length]) * factor
                for factor in self.properties["COM"]["scaling_factor"]
            ]
        )
        g = 9.81 * 1000

        timestamps = pose_sequence[0, 4, :]
        time_diff = np.diff(timestamps)
        com_pos = np.full((frames, 2), np.nan)
        com_segment = np.full((len(self.properties["COM"]["body_segments"]), 2, frames), np.nan)

        for idx, segment in enumerate(self.properties["COM"]["body_segments"]):
            segment_indices = [
                kpt_id
                for kpt in segment
                for kpt_id, kpt_info in self.halpe26["keypoint_info"].items()
                if kpt_info["name"] == kpt
            ]
            segment_pos = np.array(self.properties["COM"]["segment_pos"][idx])
            segment_data = pose_sequence[segment_indices, 1:3, :]

            for frame in range(frames):
                if not np.isnan(segment_data[:, :, frame]).all():
                    com_segment[idx, :, frame] = segment_data[0, :, frame] + segment_pos * (
                        segment_data[1, :, frame] - segment_data[0, :, frame]
                    )

        for frame in range(frames):
            if not np.isnan(com_segment[:, :, frame]).all():
                com_pos[frame, :] = np.average(
                    com_segment[:, :, frame],
                    axis=0,
                    weights=self.properties["COM"]["body_segments_mass"],
                )

        com_vel = np.diff(com_pos, axis=0) / time_diff[:, np.newaxis]
        com_pos = com_pos[1:, :]
        xcom = com_pos + (com_vel / np.sqrt(g / l))
        return np.nanmean(xcom, axis=0) if not np.isnan(xcom).all() else np.full((2,), np.nan)

    def stability_detector(self, bos, xcom):
        """
        Determines whether the xCOM is within the BOS.

        Args:
            bos (np.ndarray): Base of Support.
            xcom (np.ndarray): Extrapolated Center of Mass.

        Returns:
            bool: True if stable, False otherwise.
        """
        if bos[~np.isnan(bos).any(axis=1), :].shape[0] > 2 and np.any(~np.isnan(xcom)):
            bos = bos[~np.isnan(bos).any(axis=1), :]
            hull = ConvexHull(bos)
            point = Point(xcom)
            polygon = Polygon(bos[hull.vertices, :])
            return polygon.contains(point)
        return False

    def detect_steps(self, pose_sequence, fps):
        """
        Detect steps based on the velocity of the ankle keypoints.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.

        Returns:
            tuple: step_time (float), nsteps (int), step_events (np.ndarray)
        """
        # Extract timestamps and coordinates for the ankle keypoints
        timestamps = pose_sequence[0, 0, :]
        coordinates = pose_sequence[self.ankle_kpts, 1:4, :]
        time_diff = np.diff(timestamps)

        # Compute velocity and its magnitude
        velocity = np.diff(coordinates, axis=2) / time_diff
        velocity_magnitude = np.linalg.norm(velocity, axis=1)

        # Identify steps based on the velocity threshold
        step_array = velocity_magnitude > self.step_vel_thr
        step_events = []
        column = 0
        mode = True
        row = []

        # Detect step events
        while column < step_array.shape[1]:
            if mode:
                idx = np.argwhere(step_array[:, column:])
                if len(idx) > 0:
                    idx = idx[np.argsort(idx[:, 1])]
                    column += idx[0][1]
                    row = idx[0][0]
                    mode = False
                else:
                    break
            else:
                idx = np.argwhere(np.invert(step_array[row, column:]))
                if len(idx) > 0:
                    step_init = column
                    column += idx[0][0]
                    step_end = column
                    step_duration = timestamps[step_end + 1] - timestamps[step_init + 1]
                    step_length = np.linalg.norm(
                        coordinates[row, 0:2, step_end] - coordinates[row, 0:2, step_init]
                    )
                    step_direction = coordinates[row, 1, step_end] - coordinates[row, 1, step_init]
                    if step_length > self.step_len_thr and step_direction > 0:
                        step_events.append(
                            [row, timestamps[step_end + 1], step_duration, step_length]
                        )
                    row = []
                    mode = True
                else:
                    break

        # Convert step events to a NumPy array
        step_events = np.asarray(step_events)

        # Determine step time and number of steps
        if step_events.any():
            step_time = step_events[0][1]
            nsteps = step_events.shape[0]
        else:
            step_time = None
            nsteps = 0

        return step_time, nsteps, step_events

    def detect_pull(self, pose_sequence, fps, step_time=None):
        """
        Detect a pull event based on the pose sequence.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.
            step_time (float, optional): Time of the first detected step. Defaults to None.

        Returns:
            tuple: pull_time (float), pull_magn (float)
        """
        # Extract timestamps and coordinates for the shoulder keypoints
        timestamps = pose_sequence[0, 0, :]
        coordinates_raw = pose_sequence[self.shoulder_kpts, 1:4, :]
        time_diff = np.diff(timestamps)

        # Smooth the shoulder keypoints
        coordinates = self.smooth_keypoints(coordinates_raw)

        # Compute velocity and acceleration
        velocity = np.diff(coordinates, axis=2) / time_diff
        acceleration = np.diff(velocity, axis=2) / time_diff[1:]
        acceleration = np.nanmean(acceleration[:, 1, :], axis=0)

        # Compute baseline acceleration statistics
        baseline = int(fps * self.baseline_win)
        base_acc = acceleration[:baseline]
        base_acc_mean = np.nanmean(base_acc[base_acc > 0])
        base_acc_std = np.nanstd(base_acc[base_acc > 0])
        threshold = base_acc_mean + 3 * base_acc_std

        # Detect pull onset
        pull = acceleration[baseline:] > threshold
        if pull.any():
            if step_time:
                stop = np.where(timestamps == step_time)[0][0]
            else:
                stop = len(acceleration)
            peaks, _ = find_peaks(acceleration[baseline:stop])
            peaks = peaks + baseline
            if peaks.any():
                pull_magn = np.max(acceleration[peaks])
                pull_peak_time = peaks[np.argmax(acceleration[peaks])]
                # Get pull onset time = last minimum before pull peak
                min_peaks, _ = find_peaks(-1 * acceleration[:pull_peak_time])
                if len(min_peaks) > 0:
                    pull_onset = min_peaks[-1]
                    pull_time = timestamps[pull_onset + 2]
                else:
                    pull_time = None
            else:
                pull_time = None
                pull_magn = None
        else:
            pull_time = None
            pull_magn = None
        if pull_time is not None:
            print("pull detected")
        return pull_time, pull_magn

    def update(self, pose_sequence, fps):
        """
        Update the PullTestMonitor with the latest pose sequence.

        Args:
            pose_sequence (np.ndarray): Sequence of poses (keypoints).
            fps (float): Frames per second.

        Returns:
            tuple: BOS, xCOM, stability, baseline_met, step_time, nsteps, step_events, pull_time, pull_magn
        """
        required_frames = int(fps * self.baseline_win)

        # Default return values
        BOS = np.full((8, 2), np.nan)
        xCOM = np.full((2,), np.nan)
        stability = False
        baseline_met = False
        step_time = None
        nsteps = 0
        step_events = np.array([])
        pull_time = None
        pull_magn = None

        # Check if there are enough frames for processing
        if pose_sequence.shape[2] < required_frames or np.isnan(pose_sequence).all():
            return (
                BOS,
                xCOM,
                stability,
                baseline_met,
                step_time,
                nsteps,
                step_events,
                pull_time,
                pull_magn,
            )

        # Smooth the pose sequence once before computations
        smoothed_pose_sequence = self.smooth_keypoints(pose_sequence)

        # Perform computations using the smoothed pose sequence
        BOS = self.compute_BOS(smoothed_pose_sequence, fps)
        xCOM = self.compute_xCOM(smoothed_pose_sequence, fps)
        stability = self.stability_detector(BOS, xCOM)
        baseline_met = self.ensure_baseline(smoothed_pose_sequence, fps)

        if baseline_met:
            step_time, nsteps, step_events = self.detect_steps(smoothed_pose_sequence, fps)
            pull_time, pull_magn = self.detect_pull(smoothed_pose_sequence, fps, step_time)

        return (
            BOS,
            xCOM,
            stability,
            baseline_met,
            step_time,
            nsteps,
            step_events,
            pull_time,
            pull_magn,
        )


class RealTimeViewer(QMainWindow):
    def __init__(self, info):
        super().__init__()
        self.info = info
        self.isopen = False

        # Main window settings
        self.setWindowTitle("3D Skeleton Viewer")
        self.setGeometry(100, 100, 1600, 400)  # Width for three visualizations

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Add frames for each panel to ensure equal spacing
        skeleton_frame = QFrame()
        skeleton_layout = QVBoxLayout()
        skeleton_frame.setLayout(skeleton_layout)
        layout.addWidget(skeleton_frame, stretch=1)  # Add Skeleton Viewer frame

        bos_xcom_frame = QFrame()
        bos_xcom_layout = QVBoxLayout()
        bos_xcom_frame.setLayout(bos_xcom_layout)
        layout.addWidget(bos_xcom_frame, stretch=1)  # Add BOS and xCOM Viewer frame

        threshold_frame = QFrame()
        threshold_layout = QVBoxLayout()
        threshold_frame.setLayout(threshold_layout)
        layout.addWidget(threshold_frame, stretch=1)  # Add Threshold Fit Viewer frame

        # 1. Skeleton Viewer
        self.skeleton_viewer = GLViewWidget()
        self.skeleton_viewer.opts["xRange"] = [-1, 1]
        self.skeleton_viewer.opts["yRange"] = [0, 5]
        self.skeleton_viewer.opts["zRange"] = [-1, 2]
        self.skeleton_viewer.setCameraPosition(distance=1.5, elevation=10, azimuth=-90)
        skeleton_layout.addWidget(self.skeleton_viewer)  # Add Skeleton Viewer to its frame

        # FPS display in the Skeleton Viewer
        self.fps_label = QLabel("FPS: 0", self.skeleton_viewer)
        self.fps_label.setStyleSheet(
            "color: white; font-size: 16px; background-color: black; padding: 5px;"
        )
        self.fps_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.fps_label.setGeometry(10, 10, 100, 30)  # Position and size of the label

        # 2. BOS and xCOM Viewer (2D Plot)
        self.bos_xcom_viewer = PlotWidget()
        self.bos_xcom_viewer.setTitle("BOS and xCOM")
        self.bos_xcom_viewer.setLabel("left", "Y")
        self.bos_xcom_viewer.setLabel("bottom", "X")
        self.bos_xcom_viewer.setAspectLocked(True)  # Keep aspect ratio 1:1
        self.bos_xcom_viewer.setXRange(-2, 2)
        self.bos_xcom_viewer.setYRange(-2, 2)
        bos_xcom_layout.addWidget(self.bos_xcom_viewer)  # Add BOS and xCOM Viewer to its frame

        # Initialize BOS and xCOM plot elements with brighter colors
        self.bos_left_plot = ScatterPlotItem(size=5, brush="yellow")  # Bright pink
        self.bos_right_plot = ScatterPlotItem(size=5, brush="yellow")  # Lighter pink
        self.bos_polygon_plot = PlotDataItem(
            pen={"color": "magenta", "width": 2}, fillLevel=0, brush=(255, 105, 180, 50)
        )  # Pink with transparency
        self.xcom_plot = ScatterPlotItem(size=10, brush="white")  # Bright green

        # Add elements to the BOS and xCOM viewer
        self.bos_xcom_viewer.addItem(self.bos_left_plot)
        self.bos_xcom_viewer.addItem(self.bos_right_plot)
        self.bos_xcom_viewer.addItem(self.bos_polygon_plot)
        self.bos_xcom_viewer.addItem(self.xcom_plot)

        # 3. Threshold Fit Viewer (3D Placeholder)
        self.threshold_fit_viewer = GLViewWidget()
        self.threshold_fit_viewer.opts["xRange"] = [-1, 1]
        self.threshold_fit_viewer.opts["yRange"] = [0, 5]
        self.threshold_fit_viewer.opts["zRange"] = [-1, 2]
        self.threshold_fit_viewer.setCameraPosition(distance=1.5, elevation=10, azimuth=-90)
        threshold_layout.addWidget(
            self.threshold_fit_viewer
        )  # Add Threshold Fit Viewer to its frame

        # Show the window
        self.show()
        self.isopen = True

        # Initialize scatter and line objects for the Skeleton Viewer
        self.scatters = {}
        self.lines = {}

        for kpt_id, kpt in info["keypoint_info"].items():
            scatter = GLScatterPlotItem(
                pos=np.zeros((1, 3)), size=5, color=np.array(kpt["color"]) / 255.0, pxMode=True
            )
            self.skeleton_viewer.addItem(scatter)
            self.scatters[kpt_id] = scatter

        for idx, link in info["skeleton_info"].items():
            line = GLLinePlotItem(
                pos=np.zeros((2, 3)), color=(1, 1, 1, 0.5), width=1, antialias=True
            )
            self.skeleton_viewer.addItem(line)
            self.lines[idx] = line

    def update(self, pose, fps, bos=None, xcom=None, stability=None, baseline_met=False):
        """
        Update the viewer with the latest pose and additional data.

        Args:
            pose (np.ndarray): Latest pose data.
            fps (float): Frames per second.
            bos (np.ndarray, optional): Base of Support. Defaults to None.
            xcom (np.ndarray, optional): Extrapolated Center of Mass. Defaults to None.
            stability (bool, optional): Stability status. Defaults to None.
            baseline_met (bool, optional): Whether the baseline condition is met. Defaults to False.
        """
        # Set alpha based on baseline_met
        alpha = 1.0 if baseline_met else 0.25

        # Update the keypoints in the Skeleton Viewer
        for kpt_id, scatter in self.scatters.items():
            if np.isnan(pose[kpt_id, 1:4]).any():
                scatter.setData(pos=np.array([[0, 0, 0]]), color=np.array([[0, 0, 0, 0]]))
            else:
                scatter.setData(
                    pos=np.array([pose[kpt_id, 1:4]]),
                    color=np.append(
                        np.array(self.info["keypoint_info"][kpt_id]["color"]) / 255.0, alpha
                    ),
                )

        # Update the lines in the Skeleton Viewer
        for idx, link in self.info["skeleton_info"].items():
            kpt_names = link["link"]
            id1 = next(
                i for i, v in self.info["keypoint_info"].items() if v["name"] == kpt_names[0]
            )
            id2 = next(
                i for i, v in self.info["keypoint_info"].items() if v["name"] == kpt_names[1]
            )

            if np.isnan(pose[id1, 1:4]).any() or np.isnan(pose[id2, 1:4]).any():
                pts = np.array([[0, 0, 0], [0, 0, 0]])
                self.lines[idx].setData(pos=pts, color=(0, 0, 0, 0))
            else:
                pts = np.array([pose[id1, :3], pose[id2, :3]])
                self.lines[idx].setData(pos=pts, color=(1, 1, 1, alpha))

        # Update BOS and xCOM plot
        if bos is not None and xcom is not None:
            # Check if BOS contains valid data
            if not np.isnan(bos).all():
                bos = bos[~np.isnan(bos).any(axis=1), :]  # Remove rows with NaN values
                if len(bos) > 2:
                    # Compute the convex hull for the polygon
                    hull = ConvexHull(bos)
                    hull_vertices = np.append(
                        hull.vertices, hull.vertices[0]
                    )  # Close the polygon by appending the first vertex
                    self.bos_polygon_plot.setData(
                        bos[hull_vertices, 0],
                        bos[hull_vertices, 1],
                        pen=None,
                        brush=(255, 105, 180, int(alpha * 255)),  # Apply alpha to the polygon
                    )
                else:
                    self.bos_polygon_plot.setData([], [])

                # Plot the BOS points
                self.bos_left_plot.setData(
                    bos[: len(bos) // 2, 0],
                    bos[: len(bos) // 2, 1],
                    brush=(255, 20, 147, int(alpha * 255)),  # Apply alpha to left BOS points
                )
                self.bos_right_plot.setData(
                    bos[len(bos) // 2 :, 0],
                    bos[len(bos) // 2 :, 1],
                    brush=(255, 20, 147, int(alpha * 255)),  # Apply alpha to right BOS points
                )

                # Dynamically adjust axis limits
                bos_center = np.nanmean(bos, axis=0)  # Compute the average center of BOS points
                x_center, y_center = bos_center[0], bos_center[1]
                x_min = np.round(x_center - 0.2)
                x_max = np.round(x_center + 0.2)
                y_min = np.round(y_center - 0.2)
                y_max = np.round(y_center + 0.2)
                self.bos_xcom_viewer.setXRange(x_min, x_max)
                self.bos_xcom_viewer.setYRange(y_min, y_max)
            else:
                # Clear BOS plots if data is invalid
                self.bos_polygon_plot.setData([], [])
                self.bos_left_plot.setData([], [])
                self.bos_right_plot.setData([], [])

            # Check if xCOM contains valid data
            if not np.isnan(xcom).all():
                # Set xCOM color based on stability
                xcom_color = (
                    (255, 255, 255, int(alpha * 255))
                    if stability
                    else (255, 0, 0, int(alpha * 255))
                )
                self.xcom_plot.setData(
                    [xcom[0]], [xcom[1]], brush=xcom_color  # Apply color based on stability
                )
            else:
                # Clear xCOM plot if data is invalid
                self.xcom_plot.setData([], [])

        # Update the view
        self.skeleton_viewer.update()
        self.threshold_fit_viewer.update()
        QApplication.processEvents()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.isopen = False
            self.close()


class Custom:

    def __init__(
        self,
        det_class: str = None,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose_class: str = None,
        pose: str = None,
        pose_input_size: tuple = (192, 256),
        mode: str = None,
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cuda",
    ):

        if det_class is not None:
            try:
                det_class = getattr(rtmlib_module, det_class)
                self.det_model = det_class(
                    det, model_input_size=det_input_size, backend=backend, device=device
                )
                self.one_stage = False

            except ImportError:
                raise ImportError(f"{det_class} is not supported by rtmlib.")
        else:
            self.one_stage = True

        if pose_class is not None:
            try:
                pose_class = getattr(rtmlib_module, pose_class)
                self.pose_model = pose_class(
                    pose,
                    model_input_size=pose_input_size,
                    to_openpose=to_openpose,
                    backend=backend,
                    device=device,
                )
            except ImportError:
                raise ImportError(f"{pose_class} is not supported by rtmlib.")

    MODE = {
        "performance": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",  # noqa
            "pose_input_size": (288, 384),
        },
        "lightweight": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",  # noqa
            "det_input_size": (416, 416),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
    }

    RTMO_MODE = {
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
    }

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
