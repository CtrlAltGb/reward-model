"""
Pre-process Manav dual-arm MCAP episodes to _cached.npz files.

Reads native ROS2 CDR-encoded MCAP topics and produces a chunked npz format
for the manav_chunked_transform (state[T',14], action[T',140], is_first,
is_last, observation.images.cam_head), where T' = T - chunk_size + 1.

State layout (14 dims, at chunk-start frame):
    [0:7]   left arm joint positions from /manav/joint_states
            (la1, la2, la3, la4, la5, la6, la7)
    [7:14]  left wrist pose (xyz + qxyzw) from /manav/teleop/target

Action layout (chunk_size * 14 = 140 dims, time-first flattened):
    chunk[j][0:6]   left hand joint commands (chunk_size × 6 = 60 total)
    chunk[j][6]     left gripper             (chunk_size × 1 = 10 total)
    chunk[j][7:14]  left wrist target pose   (chunk_size × 7 = 70 total)

NaN handling: /manav/teleop/target messages at episode start are NaN (teleop
controller not yet active). Only valid (non-NaN) messages are used for
nearest-neighbour sync. Frames that cannot be synced to any valid message
are skipped.

Frame sync: /manav/cameras/head_cam/frame_index (10 fps)
Subsampled: every other frame → 5 fps

Usage:
    /data/.conda/envs/openx/bin/python3 scripts/preprocess_manav_episodes.py \\
        --root deminf_data/002 --splits train_sel test_sel --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/src")

SUBSAMPLE = 2  # 10 fps → 5 fps

# Left arm joint indices within /manav/joint_states position array
# Order in joint_states: la1, neck, ra1, la2, head, ra2, la3, ra3, la4, ra4, la5, ra5, la6, ra6, la7, ra7
_LA_INDICES = [0, 3, 6, 8, 10, 12, 14]  # la1..la7


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
    import math

    import numpy as np

    ep_name = os.path.basename(ep_dir)
    cache = os.path.join(ep_dir, ep_name + "_cached.npz")
    if os.path.exists(cache):
        return ep_dir, 0.0, True

    t0 = time.time()

    mcap_path = os.path.join(ep_dir, ep_name + "_0.mcap")
    if not os.path.exists(mcap_path):
        mcap_path = os.path.join(ep_dir, ep_name + ".mcap")
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"No MCAP found in {ep_dir}")

    from mcap_ros2.reader import read_ros2_messages

    TOPICS = [
        "/manav/cameras/head_cam/frame_index",
        "/manav/joint_states",
        "/manav/hands/left/command",
        "/manav/teleop/target",
    ]

    buckets: dict[str, list] = {t: [] for t in TOPICS}
    for msg in read_ros2_messages(mcap_path, topics=TOPICS):
        lt = msg.log_time
        t_s = lt.hour * 3600 + lt.minute * 60 + lt.second + lt.microsecond * 1e-6
        buckets[msg.channel.topic].append((t_s, msg.ros_msg))

    fi_msgs = sorted(buckets["/manav/cameras/head_cam/frame_index"], key=lambda x: x[1].frame_number)
    if not fi_msgs:
        raise RuntimeError(f"No frame_index messages in {ep_dir}")

    # Build lookup lists
    def _times(topic):
        return [x[0] for x in buckets[topic]]

    def _msgs(topic):
        return [x[1] for x in buckets[topic]]

    js_times = _times("/manav/joint_states")
    js_msgs = _msgs("/manav/joint_states")

    lh_times = _times("/manav/hands/left/command")
    lh_msgs = _msgs("/manav/hands/left/command")

    # Pre-filter teleop/target to valid (non-NaN) messages only
    tt_valid = [
        (t, m) for t, m in buckets["/manav/teleop/target"]
        if not math.isnan(m.left.wrist_pose.position.x) and not math.isnan(m.left.gripper)
    ]
    if not tt_valid:
        raise RuntimeError(f"No valid (non-NaN) /manav/teleop/target messages in {ep_dir}")
    tt_times_v = [x[0] for x in tt_valid]
    tt_msgs_v = [x[1] for x in tt_valid]

    def _get_la(t):
        if not js_msgs:
            return np.zeros(7, dtype=np.float32)
        m = js_msgs[_nearest(js_times, t)]
        pos = list(m.position)
        return np.array([pos[i] for i in _LA_INDICES], dtype=np.float32)

    def _get_lhand(t):
        if not lh_msgs:
            return np.zeros(6, dtype=np.float32)
        m = lh_msgs[_nearest(lh_times, t)]
        return np.array(list(m.positions)[:6], dtype=np.float32)

    def _get_teleop(t):
        """Return (left_gripper, left_wrist_pose[7]) from nearest valid teleop/target."""
        m = tt_msgs_v[_nearest(tt_times_v, t)]
        lg = float(m.left.gripper)
        wp = m.left.wrist_pose
        wrist = np.array([
            wp.position.x, wp.position.y, wp.position.z,
            wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w,
        ], dtype=np.float32)
        return lg, wrist

    # Subsample frames
    all_fi = sorted(fi_msgs, key=lambda x: x[1].frame_number)
    fi_sub = all_fi[::SUBSAMPLE]

    # Only keep frames within the valid teleop/target time window
    t_teleop_start = tt_times_v[0]
    fi_sub = [(t, m) for t, m in fi_sub if t >= t_teleop_start]
    if not fi_sub:
        raise RuntimeError(f"No frames within valid teleop window in {ep_dir}")

    states = []
    actions = []
    for t_s, _ in fi_sub:
        la = _get_la(t_s)
        lh = _get_lhand(t_s)
        lg, wp = _get_teleop(t_s)

        state = np.concatenate([la, wp]).astype(np.float32)    # (14,)
        action = np.concatenate([lh, [lg], wp]).astype(np.float32)  # (14,)

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
        state_at_chunk.append(states[t])                # (14,)

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
