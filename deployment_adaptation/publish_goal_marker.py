#!/usr/bin/env python3
"""Standalone helper node: watches /goal_pose and republishes a fixed
SPHERE marker at the exact same world coordinates, for RViz visualization.

Written as a separate script rather than touching rl_nav_controller.py -
zero risk to the actual deployment/navigation code, since this only
subscribes to a topic that already exists and publishes a new one that
nothing else depends on. Kept in sru_retrain/deployment_adaptation/ per
project convention (staged area), not copied into the live
sru-robot-deployment package since it's a debugging/visualization tool,
not part of the actual navigation pipeline.

Frame: publishes in whatever frame_id the incoming /goal_pose message
itself used (we've been publishing goals with frame_id='ground_truth_odom'
throughout this session's tests) - so it automatically matches whatever
frame you actually set the goal in, no hardcoding.

Durability: TRANSIENT_LOCAL, so a late-joining RViz subscriber (e.g.
launched after the goal was already set) still sees the last marker
immediately, rather than needing a fresh goal to be republished.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class GoalMarkerPublisher(Node):
    def __init__(self):
        super().__init__('goal_marker_publisher')

        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_pub = self.create_publisher(Marker, '/vis/manual_goal_marker', marker_qos)
        self.sub = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.get_logger().info('goal_marker_publisher ready - waiting for /goal_pose')

    def on_goal(self, msg: PoseStamped):
        marker = Marker()
        marker.header = msg.header  # same frame_id/stamp the goal itself was published with
        marker.ns = 'manual_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = msg.pose
        # Was 0.5 - increased per explicit request (goal marker wasn't
        # noticed/visible enough against the 30x30m terrain scale).
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.9
        self.marker_pub.publish(marker)
        self.get_logger().info(
            f'Goal marker placed at x={msg.pose.position.x:.3f}, '
            f'y={msg.pose.position.y:.3f}, frame={msg.header.frame_id}'
        )


def main():
    rclpy.init()
    node = GoalMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
