"""
Pre-process Manav dual-arm MCAP episodes to _cached.npz files.

Reads native ROS2 CDR-encoded MCAP topics and produces a chunked npz format
for the manav_chunked_transform (state[T',14], action[T',140], is_first,
is_last, observation.images.cam_head), where T' = T - chunk_size + 1.

State layout (14 dims, at chunk-start frame):
    [0:7]   left arm joint positions from /manav/joint_states (la1..la7 by name)
    [7:14]  left wrist pose (xyz + qxyzw) from teleop topic

Action layout (chunk_size * 14 = 140 dims, time-first flattened):
    chunk[j][0:6]   left hand joint commands
    chunk[j][6]     left gripper
    chunk[j][7:14]  left wrist target pose

Two topic layouts are supported (auto-detected):

  Old Manav layout (task_2_data_real):
    /manav/hands/left/command  → hand positions
    /manav/teleop/target       → wrist_pose + gripper
    Joint names: la1, la2, ... (no prefix)

  New raybot layout (real_world_data_filtered):
    /manav/joint/command/gated → arm + hand positions
    /manav/teleop/control      → wrist_pose + grip
    Joint names: joint_la1, joint_la2, ... (joint_ prefix)

Frame sync: /manav/cameras/head_cam/frame_index (10 fps)
Subsampled: every other frame → 5 fps
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/src")

SUBSAMPLE = 2  # 10 fps → 5 fps


def _nearest(timestamps: list[float], t: float) -> int:
    """Return index of closest timestamp to t."""
    best = 0
    best_d = abs(timestamps[0] - t)
    for i, ts in enumerate(timestamps[1:], 1):
        d = abs(ts - t)
        if d < best_d:
            best_d = d
            best = i
    return best


def _la_indices_from_names(names: list[str]) -> list[int]:
    """Return indices for la1..la7 in joint_states, handling both 'la1' and 'joint_la1'."""
    found = {}
    for i, name in enumerate(names):
        m = re.search(r'la(\d+)', name)
        if m:
            k = int(m.group(1))
            if 1 <= k <= 7:
                found[k] = i
    return [found[k] for k in sorted(found)]


def _left_hand_indices_from_names(names: list[str]) -> list[int]:
    """Return indices for 6 left finger joints in /manav/joint/command/gated."""
    order = [
        "left_little_1_joint",
        "left_ring_1_joint",
        "left_middle_1_joint",
        "left_index_1_joint",
        "left_thumb_2_joint",
        "left_thumb_1_joint",
    ]
    return [names.index(n) for n in order if n in names]


def process_one(ep_dir: str) -> tuple:
    import math

    import numpy as np

    ep_name = os.path.basename(ep_dir)
    cache = os.path.join(ep_dir, ep_name + "_cached.npz")
    if os.path.exists(cache):
        return ep_dir, 0.0, True

    t0 = time.time()

    # Resolve MCAP path (handles both {ep}_0.mcap and session_name glob)
    mcap_path = os.path.join(ep_dir, ep_name + "_0.mcap")
    if not os.path.exists(mcap_path):
        mcap_path = os.path.join(ep_dir, ep_name + ".mcap")
    if not os.path.exists(mcap_path):
        import glob as _glob
        matches = _glob.glob(os.path.join(ep_dir, "*_0.mcap"))
        mcap_path = matches[0] if matches else ""
    if not mcap_path or not os.path.exists(mcap_path):
        raise FileNotFoundError(f"No MCAP found in {ep_dir}")

    from mcap_ros2.reader import read_ros2_messages

    ALL_TOPICS = [
        "/manav/cameras/head_cam/frame_index",
        "/manav/joint_states",
        # Old Manav layout
        "/manav/hands/left/command",
        "/manav/teleop/target",
        # New raybot layout
        "/manav/teleop/control",
        "/manav/joint/command/gated",
    ]

    buckets: dict[str, list] = {t: [] for t in ALL_TOPICS}
    for msg in read_ros2_messages(mcap_path, topics=ALL_TOPICS):
        lt = msg.log_time
        t_s = lt.hour * 3600 + lt.minute * 60 + lt.second + lt.microsecond * 1e-6
        buckets[msg.channel.topic].append((t_s, msg.ros_msg))

    fi_msgs = sorted(buckets["/manav/cameras/head_cam/frame_index"], key=lambda x: x[1].frame_number)
    if not fi_msgs:
        raise RuntimeError(f"No frame_index messages in {ep_dir}")

    # /manav/joint_states carries three message variants on the same topic:
    # full arm (16 joints, includes la*), left hand (6 joints), right hand (6 joints).
    # Filter to arm-only messages so hand messages don't corrupt la* index lookup.
    arm_msgs = [(t, m) for t, m in buckets["/manav/joint_states"]
                if any(re.search(r'la\d', n) for n in m.name)]
    if not arm_msgs:
        raise RuntimeError(f"No arm joint_states messages in {ep_dir}")

    js_times = [x[0] for x in arm_msgs]
    js_ros = [x[1] for x in arm_msgs]

    # Detect layout from which teleop topic has data
    use_new_layout = len(buckets["/manav/teleop/control"]) > 0

    # --- Joint indices (name-based lookup, robust to reordering) ---
    la_idx = _la_indices_from_names(list(js_ros[0].name))
    if len(la_idx) < 7:
        raise RuntimeError(f"Could not find la1..la7 in joint_states names: {list(js_ros[0].name)}")

    def _get_la(t: float):
        m = js_ros[_nearest(js_times, t)]
        pos = list(m.position)
        return np.array([pos[i] for i in la_idx], dtype=np.float32)

    if use_new_layout:
        # --- New raybot layout ---
        tc_entries = buckets["/manav/teleop/control"]
        tc_valid = [
            (t, m) for t, m in tc_entries
            if not math.isnan(m.left.wrist_pose.position.x)
        ]
        if not tc_valid:
            raise RuntimeError(f"No valid /manav/teleop/control messages in {ep_dir}")
        tc_times = [x[0] for x in tc_valid]
        tc_ros = [x[1] for x in tc_valid]

        cmd_entries = buckets["/manav/joint/command/gated"]
        if not cmd_entries:
            raise RuntimeError(f"No /manav/joint/command/gated messages in {ep_dir}")
        cmd_times = [x[0] for x in cmd_entries]
        cmd_ros = [x[1] for x in cmd_entries]
        # Resolve left hand indices once from first message
        cmd_names = list(cmd_ros[0].joint_names)
        lh_cmd_idx = _left_hand_indices_from_names(cmd_names)
        if len(lh_cmd_idx) < 6:
            raise RuntimeError(f"Could not find 6 left hand joints in {cmd_names}")

        def _get_wrist_and_grip(t: float):
            m = tc_ros[_nearest(tc_times, t)]
            wp = m.left.wrist_pose
            wrist = np.array([
                wp.position.x, wp.position.y, wp.position.z,
                wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w,
            ], dtype=np.float32)
            return float(m.left.grip), wrist

        def _get_lhand(t: float):
            m = cmd_ros[_nearest(cmd_times, t)]
            pos = list(m.positions)
            return np.array([pos[i] for i in lh_cmd_idx], dtype=np.float32)

        t_teleop_start = tc_times[0]

    else:
        # --- Old Manav layout ---
        tt_valid = [
            (t, m) for t, m in buckets["/manav/teleop/target"]
            if not math.isnan(m.left.wrist_pose.position.x) and not math.isnan(m.left.gripper)
        ]
        if not tt_valid:
            raise RuntimeError(f"No valid (non-NaN) /manav/teleop/target messages in {ep_dir}")
        tt_times = [x[0] for x in tt_valid]
        tt_ros = [x[1] for x in tt_valid]

        lh_entries = buckets["/manav/hands/left/command"]
        lh_times = [x[0] for x in lh_entries]
        lh_ros = [x[1] for x in lh_entries]

        def _get_wrist_and_grip(t: float):
            m = tt_ros[_nearest(tt_times, t)]
            wp = m.left.wrist_pose
            wrist = np.array([
                wp.position.x, wp.position.y, wp.position.z,
                wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w,
            ], dtype=np.float32)
            return float(m.left.gripper), wrist

        def _get_lhand(t: float):
            if not lh_ros:
                return np.zeros(6, dtype=np.float32)
            m = lh_ros[_nearest(lh_times, t)]
            return np.array(list(m.positions)[:6], dtype=np.float32)

        t_teleop_start = tt_times[0]

    # Subsample frames and trim to valid teleop window
    all_fi = sorted(fi_msgs, key=lambda x: x[1].frame_number)
    fi_sub = all_fi[::SUBSAMPLE]
    fi_sub = [(t, m) for t, m in fi_sub if t >= t_teleop_start]
    if not fi_sub:
        raise RuntimeError(f"No frames within valid teleop window in {ep_dir}")

    states = []
    actions = []
    for t_s, _ in fi_sub:
        la = _get_la(t_s)
        lh = _get_lhand(t_s)
        lg, wp = _get_wrist_and_grip(t_s)

        state = np.concatenate([la, wp]).astype(np.float32)          # (14,)
        action = np.concatenate([lh, [lg], wp]).astype(np.float32)   # (14,)

        states.append(state)
        actions.append(action)

    if not states:
        raise RuntimeError(f"No frames extracted from {ep_dir}")

    chunk_size = 10
    if len(states) < chunk_size:
        raise RuntimeError(
            f"Episode too short for chunk_size={chunk_size}: {len(states)} frames in {ep_dir}"
        )

    action_chunks = []
    state_at_chunk = []
    for t in range(len(actions) - chunk_size + 1):
        chunk = np.stack(actions[t : t + chunk_size])  # (chunk_size, 14)
        action_chunks.append(chunk.flatten())           # (140,)
        state_at_chunk.append(states[t])               # (14,)

    state_arr = np.stack(state_at_chunk)    # (T', 14)
    action_arr = np.stack(action_chunks)    # (T', 140)
    T = len(state_at_chunk)
    is_first = np.zeros(T, dtype=bool)
    is_first[0] = True
    is_last = np.zeros(T, dtype=bool)
    is_last[-1] = True

    # Decode video frames — only need frames at chunk-start positions
    mp4_path = os.path.join(ep_dir, f"{ep_name}_head_cam.mp4")
    if not os.path.exists(mp4_path):
        mp4_path = os.path.join(ep_dir, f"{ep_name}_cam_head.mp4")
    if not os.path.exists(mp4_path):
        import glob as _glob
        matches = _glob.glob(os.path.join(ep_dir, "*_head_cam.mp4"))
        mp4_path = matches[0] if matches else ""
    if not mp4_path or not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No head camera MP4 in {ep_dir}")

    import imageio.v3 as iio
    import tensorflow as tf

    def _center_crop_resize(frame, size=256):
        h, w = frame.shape[:2]
        crop = min(h, w)
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        cropped = frame[y0:y0 + crop, x0:x0 + crop]
        return tf.image.resize(cropped, [size, size]).numpy().astype(np.uint8)

    all_frames = iio.imread(mp4_path, plugin="pyav")
    fi_sub_indices = [int(x[1].frame_number) for x in fi_sub]

    jpeg_frames = []
    for fi in fi_sub_indices[:T]:   # only chunk-start positions
        frame = all_frames[fi] if fi < len(all_frames) else all_frames[-1]
        frame = _center_crop_resize(frame)
        jpeg = tf.image.encode_jpeg(frame, quality=95).numpy()
        jpeg_frames.append(jpeg)

    save_dict = {
        "state": state_arr,
        "action": action_arr,
        "is_first": is_first,
        "is_last": is_last,
        "observation.images.cam_head": np.array(jpeg_frames, dtype=object),
    }
    np.savez_compressed(cache, **save_dict)

    return ep_dir, time.time() - t0, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="episode_data")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ep_dirs = []
    for split in args.splits:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split_dir} (not found)")
            continue
        for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
            if entry.is_dir():
                ep_dirs.append(entry.path)

    total = len(ep_dirs)
    print(f"Processing {total} episodes with {args.workers} workers…\n")

    done = skipped = failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, d): d for d in ep_dirs}
        for fut in as_completed(futures):
            ep_dir = futures[fut]
            ep_name = os.path.basename(ep_dir)
            done += 1
            try:
                ep_dir, elapsed, was_cached = fut.result()
                tag = "cached" if was_cached else f"{elapsed:.1f}s"
                if was_cached:
                    skipped += 1
            except Exception as exc:
                failed += 1
                tag = f"SKIP: {exc}"
            pct = done / total * 100
            print(f"  [{done:3d}/{total}  {pct:5.1f}%]  {ep_name}  ({tag})", flush=True)

    new = done - skipped - failed
    print(f"\nDone. {new} newly cached, {skipped} already cached, {failed} skipped (errors). "
          f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
