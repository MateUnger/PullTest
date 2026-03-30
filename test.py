import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A
from rtmlib import BodyWithFeet


body_with_feet = BodyWithFeet(
    to_openpose=False, mode="performance", backend="onnxruntime", device="cuda"
)


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    while 1:
        capture = k4a.get_capture()
        color = capture.color
        if np.any(color):
            color = cv2.resize(color, (color.shape[1] // 4, color.shape[0] // 4))
            keypoints, scores = body_with_feet(color)

            cv2.imshow("k4a", capture.color[:, :, :3])
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.stop()


if __name__ == "__main__":
    main()
