#!/usr/bin/env python3
"""One-shot check: subscribe to the depth topic, grab one frame, print real
statistics (min/max/mean, NaN/Inf/zero counts) - not just the encoding type."""

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthChecker(Node):
    def __init__(self):
        super().__init__('depth_checker')
        self.bridge = CvBridge()
        self.got_one = False
        self.sub = self.create_subscription(
            Image, '/zed/zed_node/depth/depth_registered', self.cb, 10
        )

    def cb(self, msg):
        if self.got_one:
            return
        self.got_one = True
        arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        arr = np.asarray(arr, dtype=np.float32)
        print(f"shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"min: {np.nanmin(arr):.4f}  max: {np.nanmax(arr):.4f}  mean: {np.nanmean(arr):.4f}")
        n = arr.size
        print(f"NaN count: {np.isnan(arr).sum()} / {n} ({100*np.isnan(arr).sum()/n:.1f}%)")
        print(f"Inf count: {np.isinf(arr).sum()} / {n} ({100*np.isinf(arr).sum()/n:.1f}%)")
        print(f"exactly-zero count: {(arr == 0.0).sum()} / {n} ({100*(arr == 0.0).sum()/n:.1f}%)")
        print(f"in-range [0.25, 10.0]m count: {((arr >= 0.25) & (arr <= 10.0)).sum()} / {n} "
              f"({100*((arr >= 0.25) & (arr <= 10.0)).sum()/n:.1f}%)")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DepthChecker()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
