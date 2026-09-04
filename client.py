"""V-JEPA WebSocket client.

Example:
    python client.py --video sample_video.mp4 --url ws://127.0.0.1:8000/ws
"""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import websockets


def encode_video_frames(video_path: str, num_frames: int = 16, quality: int = 90):
    """Read evenly spaced frames and return base64 JPEG strings."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError("Video has no readable frames")

    # The server uses tubelet_size=2, so send an even number of frames.
    num_frames = max(2, int(num_frames))
    if num_frames % 2:
        num_frames -= 1
    indices = np.linspace(0, total - 1, num_frames).round().astype(int)

    encoded = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            encoded.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
    capture.release()

    if len(encoded) < 2:
        raise RuntimeError("Could not encode enough video frames")
    if len(encoded) % 2:
        encoded.pop()
    return encoded


async def request_inference(url: str, frames, topk: int):
    # JPEG frames can make a large JSON message; increase the receive limit.
    async with websockets.connect(url, max_size=50 * 1024 * 1024) as socket:
        await socket.send(json.dumps({"frames": frames, "topk": topk}))
        return json.loads(await socket.recv())


def main():
    parser = argparse.ArgumentParser(description="Send a local video to V-JEPA server")
    parser.add_argument("--video", default="sample_video.mp4", help="local video path")
    parser.add_argument("--url", default="ws://192.168.3.5:8099/ws")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    frames = encode_video_frames(args.video, args.frames, args.jpeg_quality)
    result = asyncio.run(request_inference(args.url, frames, args.topk))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
