"""Standalone verification of the adapted LearningModel.predict() pipeline against
the real exported ONNX files - no ROS2/rclpy dependency, so it can run in any plain
Python env with onnxruntime + scipy. Mirrors the deployment repo's own
test_onnx_inference.py pattern (ROS-independent sanity check before wiring into the
actual node).

Does NOT import rl_nav_controller.py directly (that file imports rclpy at module
level, unavailable outside a ROS2 env) - reimplements just the LearningModel class's
logic inline, kept in exact sync with the adapted rl_nav_controller.py in this same
directory. If you change one, change the other.
"""

import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/joshua/sru_ws/src/sru-robot-deployment/rl_nav_controller/rl_nav_controller")
from utils import subtract_frame_transforms, transform_points  # noqa: E402

VAE_ENCODER_PATH = "/home/joshua/sru_ws/src/sru-robot-deployment/rl_nav_controller/deployment_policies/vae_encoder.onnx"
NAV_POLICY_PATH = "/home/joshua/sru_ws/src/sru_retrain/deployment_adaptation/exported_policy/nav_policy.onnx"

MAX_LINEAR_VELOCITY = 3.0
MAX_ANGULAR_VELOCITY = 2.523
LSTM_HIDDEN_DIM = 512


def depth_preprocess(sess, img):
    import cv2
    img_resized = cv2.resize(img, (64, 40), interpolation=cv2.INTER_LINEAR)
    img_tensor = img_resized.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    out = sess.run(None, {sess.get_inputs()[0].name: img_tensor})[0]
    return out.flatten()


def normalize_target_position(target_pos_w, robot_pos_w, robot_orientation_w):
    target_pos_w = np.array(target_pos_w, dtype=np.float32)[np.newaxis]
    robot_pos_w = np.array(robot_pos_w, dtype=np.float32)[np.newaxis]
    robot_orientation_w = np.array(robot_orientation_w, dtype=np.float32)[np.newaxis]
    inv_pos, inv_rot = subtract_frame_transforms(robot_pos_w, robot_orientation_w)
    target_vec_b = transform_points(target_pos_w, inv_pos, inv_rot)
    dist = np.linalg.norm(target_vec_b, axis=-1, keepdims=True) + 1e-6
    target_pos = target_vec_b / dist
    dist_log = np.log(dist + 1.0)
    target_pos = np.concatenate((target_pos, dist_log), axis=-1).reshape(-1)
    return target_pos


def main():
    print("[verify] loading real ONNX files...")
    vae_sess = ort.InferenceSession(VAE_ENCODER_PATH, providers=["CPUExecutionProvider"])
    policy_sess = ort.InferenceSession(NAV_POLICY_PATH, providers=["CPUExecutionProvider"])
    print("[verify] both sessions loaded OK.")

    h_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
    c_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)

    # Realistic-ish dummy inputs
    linear_vel = [0.2, 0.0, 0.0]
    angular_vel = [0.0, 0.0, 0.05]
    gravity_vector = [0.0, 0.0, -1.0]
    last_action = [0.0, 0.0]  # 2-dim - the whole point of this check
    target_pos_w = [3.0, 1.0, 0.0]
    robot_pos_w = [0.0, 0.0, 0.0]
    robot_orientation_w = [1.0, 0.0, 0.0, 0.0]
    depth_image = (np.random.rand(600, 960).astype(np.float32) * 5.0 + 0.5)

    depth_embedding = depth_preprocess(vae_sess, depth_image)
    print(f"[verify] depth_embedding shape: {depth_embedding.shape}")
    assert depth_embedding.shape == (2560,), "depth embedding shape mismatch"

    target_pos_log = normalize_target_position(target_pos_w, robot_pos_w, robot_orientation_w)
    print(f"[verify] target_pos_log shape: {target_pos_log.shape}, value: {target_pos_log}")
    assert target_pos_log.shape == (4,), "target_pos_log shape mismatch"

    state_input = np.array(
        linear_vel + angular_vel + gravity_vector + last_action + target_pos_log.tolist(),
        dtype=np.float32,
    )
    print(f"[verify] state_input shape: {state_input.shape}")
    assert state_input.shape == (15,), f"state_input should be 15-dim, got {state_input.shape}"

    obs = np.concatenate([state_input, depth_embedding])[np.newaxis].astype(np.float32)
    print(f"[verify] obs shape: {obs.shape}")
    assert obs.shape == (1, 2575), f"obs should be (1, 2575), got {obs.shape}"

    raw_action, h_out, c_out = policy_sess.run(
        None, {"obs": obs, "h_in": h_state, "c_in": c_state}
    )
    print(f"[verify] raw_action (unclamped): {raw_action}, shape {raw_action.shape}")
    assert raw_action.shape == (1, 2), f"actions should be (1, 2), got {raw_action.shape}"
    assert np.isfinite(raw_action).all(), "raw_action has NaN/Inf"
    assert np.isfinite(h_out).all() and np.isfinite(c_out).all(), "hidden state has NaN/Inf"

    cmd_vel = raw_action.copy()
    cmd_vel[:, 0] = np.clip(cmd_vel[:, 0], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
    cmd_vel[:, 1] = np.clip(cmd_vel[:, 1], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
    cmd_vel = cmd_vel.squeeze(0)
    print(f"[verify] cmd_vel (clamped): v={cmd_vel[0]:.4f} m/s, omega={cmd_vel[1]:.4f} rad/s")
    assert -MAX_LINEAR_VELOCITY <= cmd_vel[0] <= MAX_LINEAR_VELOCITY
    assert -MAX_ANGULAR_VELOCITY <= cmd_vel[1] <= MAX_ANGULAR_VELOCITY

    # Run a second step reusing h_out/c_out, to check recurrent state threading works
    obs2 = obs  # same obs is fine for this check - only testing state plumbing, not realism
    raw_action2, h_out2, c_out2 = policy_sess.run(
        None, {"obs": obs2, "h_in": h_out, "c_in": c_out}
    )
    print(f"[verify] step 2 raw_action: {raw_action2}")
    assert np.isfinite(raw_action2).all()
    assert not np.allclose(raw_action, raw_action2), (
        "action should generally differ step-to-step with evolving hidden state "
        "(not a hard requirement, but worth knowing if it doesn't)"
    )

    print("=== VERIFY_PREDICT_PIPELINE PASSED ===")


if __name__ == "__main__":
    main()
