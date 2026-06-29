import sys  # noqa: I001

sys.path.insert(0, "/data/demonstration-information")

import tensorflow as tf
from openx.data.utils import RobotType, StateEncoding

CHUNK_SIZE = 10
ACTION_DIM = 14


def manav_chunked_transform(ep: dict) -> dict:
    """
    Standardise a raw LeRobot episode from the manav_dual_arm robot (chunked actions).

    Input keys (produced by preprocess_manav_episodes.py):
        "observation.state"              float32[T', 14]  (chunk-start frames only)
        "action"                         float32[T', 140] (10-step chunks, time-first flattened)
        "observation.images.cam_head"    string[T']       (JPEG bytes, 256x256)
        "is_first"                       bool[T']
        "is_last"                        bool[T']

    where T' = T - CHUNK_SIZE + 1.
    """
    state = ep["observation.state"]  # [T', 14]
    action = ep["action"]            # [T', 140]

    T = tf.shape(action)[0]
    action_r = tf.reshape(action, [T, CHUNK_SIZE, ACTION_DIM])  # [T', 10, 14]

    observation = {
        "state": {
            StateEncoding.JOINT_POS: state[:, :7],  # left arm joints
            StateEncoding.MISC:      state[:, 7:],  # left wrist pose
        },
        "image": {
            "agent": ep["observation.images.cam_head"],
        },
    }

    structured_action = {
        "desired_absolute": {
            StateEncoding.JOINT_POS: tf.reshape(action_r[:, :, :6],  [T, CHUNK_SIZE * 6]),
            StateEncoding.GRIPPER:   tf.reshape(action_r[:, :, 6:7], [T, CHUNK_SIZE * 1]),
            StateEncoding.MISC:      tf.reshape(action_r[:, :, 7:],  [T, CHUNK_SIZE * 7]),
        },
    }

    return {
        "observation": observation,
        "action":      structured_action,
        "is_first":    ep["is_first"],
        "is_last":     ep["is_last"],
        "robot":       RobotType.UNKNOWN,
    }
