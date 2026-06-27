"""
Pre-process Manav dual-arm MCAP episodes to _cached.npz files.

Reads native ROS2 CDR-encoded MCAP topics and produces the same npz format
as _read_mcap_episode's cache (state[T,32], action[T,36], is_first, is_last,
observation.images.cam_head).

State layout (32 dims):
    [0:16]  joint positions from /manav/joint_states (order: la1,neck,ra1,la2,head,ra2,...,la7,ra7)
    [16]    left gripper from /manav/teleop/target
    [17]    right gripper from /manav/teleop/target
    [18:25] right wrist pose (xyz + qxyzw) from /manav/teleop/target
    [25:32] left  wrist pose (xyz + qxyzw) from /manav/teleop/target

Action layout (36 dims):
    [0:7]   left  arm commands from /l_arm_pos_controller/commands
    [7:14]  right arm commands from /r_arm_pos_controller/commands
    [14:16] head  commands     from /head_pos_controller/commands
    [16]    left  gripper      from /manav/teleop/target
    [17]    right gripper      from /manav/teleop/target
    [18:24] left  hand joints  from /manav/hands/left/command (6 joints)
    [24:36] zeros              (right hand / padding)

Frame sync: /manav/cameras/head_cam/frame_index (10 fps)
Subsampled: every other frame → 5 fps

Usage:
    /data/.conda/envs/openx/bin/python3 scripts/preprocess_manav_episodes.py \
        --root deminf_data/002 --splits train_sel test_sel --workers 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def process_one(ep_dir: str) -> tuple:
    import numpy as np

    ep_name = os.path.basename(ep_dir)
    cache = os.path.join(ep_dir, ep_name + "_cached.npz")
    if os.path.exists(cache):
        return ep_dir, 0.0, True

    t0 = time.time()

    mcap_path = os.path.join(ep_dir, ep_name + "_0.mcap")
    if not os.path.exists(mcap_path):
        # Fall back to canonical alias created by _setup_deminf_data
        mcap_path = os.path.join(ep_dir, ep_name + ".mcap")
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"No MCAP found in {ep_dir}")

    from mcap_ros2.reader import read_ros2_messages

    TOPICS = [
        "/manav/cameras/head_cam/frame_index",
        "/manav/joint_states",
        "/l_arm_pos_controller/commands",
        "/r_arm_pos_controller/commands",
        "/head_pos_controller/commands",
        "/manav/teleop/target",
        "/manav/hands/left/command",
    ]

    # Collect all messages by topic, keyed by log_time in seconds
    buckets: dict[str, list] = {t: [] for t in TOPICS}
    for msg in read_ros2_messages(mcap_path, topics=TOPICS):
        lt = msg.log_time
        t_s = lt.hour * 3600 + lt.minute * 60 + lt.second + lt.microsecond * 1e-6
        buckets[msg.channel.topic].append((t_s, msg.ros_msg))

    fi_msgs = sorted(buckets["/manav/cameras/head_cam/frame_index"], key=lambda x: x[0])
    if not fi_msgs:
        raise RuntimeError(f"No frame_index messages in {ep_dir}")

    # Build lookup lists for nearest-neighbour sync
    def _times(topic):
        return [x[0] for x in buckets[topic]]

    def _msgs(topic):
        return [x[1] for x in buckets[topic]]

    js_times = _times("/manav/joint_states")
    js_msgs = _msgs("/manav/joint_states")
    la_times = _times("/l_arm_pos_controller/commands")
    la_msgs = _msgs("/l_arm_pos_controller/commands")
    ra_times = _times("/r_arm_pos_controller/commands")
    ra_msgs = _msgs("/r_arm_pos_controller/commands")
    hd_times = _times("/head_pos_controller/commands")
    hd_msgs = _msgs("/head_pos_controller/commands")
    tt_times = _times("/manav/teleop/target")
    tt_msgs = _msgs("/manav/teleop/target")
    lh_times = _times("/manav/hands/left/command")
    lh_msgs = _msgs("/manav/hands/left/command")

    def _get_js(t):
        if not js_msgs:
            return np.zeros(16, dtype=np.float32)
        m = js_msgs[_nearest(js_times, t)]
        return np.array(list(m.position)[:16], dtype=np.float32)

    def _get_la(t):
        if not la_msgs:
            return np.zeros(7, dtype=np.float32)
        m = la_msgs[_nearest(la_times, t)]
        return np.array(list(m.data)[:7], dtype=np.float32)

    def _get_ra(t):
        if not ra_msgs:
            return np.zeros(7, dtype=np.float32)
        m = ra_msgs[_nearest(ra_times, t)]
        return np.array(list(m.data)[:7], dtype=np.float32)

    def _get_head(t):
        if not hd_msgs:
            return np.zeros(2, dtype=np.float32)
        m = hd_msgs[_nearest(hd_times, t)]
        return np.array(list(m.data)[:2], dtype=np.float32)

    def _get_teleop(t):
        """Return (left_gripper, right_gripper, r_wrist[7], l_wrist[7]) or zeros."""
        if not tt_msgs:
            return 0.0, 0.0, np.zeros(7, np.float32), np.zeros(7, np.float32)
        # Find nearest non-NaN message
        idx = _nearest(tt_times, t)
        for offset in range(len(tt_msgs)):
            i = (idx + offset) % len(tt_msgs)
            m = tt_msgs[i]
            lg = m.left.gripper
            rg = m.right.gripper
            if not (math.isnan(lg) or math.isnan(rg)):
                wp_r = m.right.wrist_pose
                wp_l = m.left.wrist_pose
                def _pose(wp):
                    vals = [wp.position.x, wp.position.y, wp.position.z,
                            wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w]
                    if any(math.isnan(v) for v in vals):
                        return np.zeros(7, np.float32)
                    return np.array(vals, dtype=np.float32)
                return float(lg), float(rg), _pose(wp_r), _pose(wp_l)
        return 0.0, 0.0, np.zeros(7, np.float32), np.zeros(7, np.float32)

    def _get_lhand(t):
        if not lh_msgs:
            return np.zeros(6, dtype=np.float32)
        m = lh_msgs[_nearest(lh_times, t)]
        return np.array(list(m.positions)[:6], dtype=np.float32)

    # Build per-frame arrays (all frames, then subsample)
    all_fi = sorted(fi_msgs, key=lambda x: x[1].frame_number)
    # Subsample every SUBSAMPLE frames (10fps → 5fps)
    fi_sub = all_fi[::SUBSAMPLE]

    states = []
    actions = []
    for t_s, _ in fi_sub:
        js = _get_js(t_s)
        la = _get_la(t_s)
        ra = _get_ra(t_s)
        hd = _get_head(t_s)
        lg, rg, wp_r, wp_l = _get_teleop(t_s)
        lh = _get_lhand(t_s)

        state = np.concatenate([
            js,                          # [0:16]  joint positions
            [lg, rg],                    # [16:18] gripper
            wp_r,                        # [18:25] right wrist
            wp_l,                        # [25:32] left wrist
        ]).astype(np.float32)
        action = np.concatenate([
            la,                          # [0:7]   left arm
            ra,                          # [7:14]  right arm
            hd,                          # [14:16] head
            [lg, rg],                    # [16:18] gripper
            lh,                          # [18:24] left hand
            np.zeros(12, np.float32),    # [24:36] right hand + padding
        ]).astype(np.float32)

        states.append(state)
        actions.append(action)

    if not states:
        raise RuntimeError(f"No frames extracted from {ep_dir}")

    state_arr = np.stack(states)   # (T, 32)
    action_arr = np.stack(actions) # (T, 36)
    T = len(states)
    is_first = np.zeros(T, dtype=bool)
    is_first[0] = True
    is_last = np.zeros(T, dtype=bool)
    is_last[-1] = True

    # Decode video frames
    mp4_path = os.path.join(ep_dir, f"{ep_name}_head_cam.mp4")
    if not os.path.exists(mp4_path):
        mp4_path = os.path.join(ep_dir, f"{ep_name}_cam_head.mp4")
    if not os.path.exists(mp4_path):
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

    all_frames = iio.imread(mp4_path, plugin="pyav")  # (N, H, W, 3)
    fi_sub_indices = [int(x[1].frame_number) for x in fi_sub]

    jpeg_frames = []
    for fi in fi_sub_indices:
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
    print(f"Processing {total} Manav episodes with {args.workers} workers…\n")

    done = skipped = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, d): d for d in ep_dirs}
        for fut in as_completed(futures):
            ep_dir, elapsed, was_cached = fut.result()
            ep_name = os.path.basename(ep_dir)
            done += 1
            tag = "cached" if was_cached else f"{elapsed:.1f}s"
            if was_cached:
                skipped += 1
            pct = done / total * 100
            print(f"  [{done:3d}/{total}  {pct:5.1f}%]  {ep_name}  ({tag})", flush=True)

    new = done - skipped
    print(f"\nDone. {new} newly cached, {skipped} already cached. "
          f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
