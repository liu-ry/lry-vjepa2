"""Minimal V-JEPA 2 WebSocket inference server.

The client sends JSON: {"frames": [base64_encoded_jpeg, ...], "topk": 5}.
Frames are RGB JPEG/PNG images. The server returns SSv2 predictions and the
backbone feature shape. Keep the process alive so the checkpoint is loaded once.
"""

import argparse
import base64
import io
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

app = FastAPI(title="V-JEPA 2 inference server")
MODEL = None
CLASSIFIER = None
DEVICE = None
DTYPE = torch.float16
LABELS = {}
SIZE = 384
NUM_FRAMES = 64
TUBELET_SIZE = 2
logger = logging.getLogger(__name__)


def _load_state(path, key):
    state = torch.load(path, map_location="cpu", weights_only=True)[key]
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}


def load_models(model_path, classifier_path, labels_path=None, device="cuda:0", num_frames=64, size=384):
    global MODEL, CLASSIFIER, DEVICE, LABELS, SIZE, NUM_FRAMES
    SIZE, NUM_FRAMES = size, num_frames
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = DTYPE if DEVICE.type == "cuda" else torch.float32
    MODEL = vit_giant_xformers_rope(img_size=(size, size), num_frames=num_frames, use_sdpa=True)
    msg = MODEL.load_state_dict(_load_state(model_path, "encoder"), strict=False)
    if msg.missing_keys:
        raise RuntimeError(f"encoder checkpoint is incompatible; missing {len(msg.missing_keys)} keys")
    MODEL = MODEL.to(DEVICE, dtype=dtype).eval()
    CLASSIFIER = AttentiveClassifier(embed_dim=MODEL.embed_dim, num_heads=16, depth=4, num_classes=174)
    classifier_state = torch.load(classifier_path, map_location="cpu", weights_only=True)["classifiers"][0]
    classifier_state = {k.replace("module.", ""): v for k, v in classifier_state.items()}
    msg = CLASSIFIER.load_state_dict(classifier_state, strict=False)
    if msg.missing_keys:
        raise RuntimeError(f"classifier checkpoint is incompatible; missing {len(msg.missing_keys)} keys")
    CLASSIFIER = CLASSIFIER.to(DEVICE, dtype=dtype).eval()
    if labels_path and Path(labels_path).exists():
        LABELS = json.loads(Path(labels_path).read_text())


def decode_frames(encoded_frames, size):
    frames = []
    for encoded in encoded_frames:
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        short = int(round(256 / 224 * size))
        scale = short / min(image.size)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
        left, top = (resized.width - size) // 2, (resized.height - size) // 2
        image = resized.crop((left, top, left + size, top + size))
        frames.append(torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3))
    # V-JEPA expects [B, C, T, H, W], normalized ImageNet input.
    video = torch.stack(frames).permute(3, 0, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    return ((video - mean) / std).unsqueeze(0)


@app.websocket("/ws")
async def websocket_inference(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            request = await ws.receive_json()
            frames = request.get("frames", [])
            if not frames:
                await ws.send_json({"error": "frames must be a non-empty base64 JPEG/PNG list"})
                continue
            if MODEL is None or CLASSIFIER is None:
                await ws.send_json({"error": "models are not loaded"})
                continue
            if len(frames) < TUBELET_SIZE or len(frames) % TUBELET_SIZE:
                await ws.send_json({"error": f"frames count must be a positive multiple of {TUBELET_SIZE}"})
                continue
            with torch.inference_mode():
                videos = [decode_frames(frames[i:i + NUM_FRAMES], SIZE)
                          for i in range(0, len(frames), NUM_FRAMES)]
                if any(v.shape[2] != NUM_FRAMES for v in videos):
                    await ws.send_json({"error": f"send exactly {NUM_FRAMES} or a multiple of {NUM_FRAMES} frames"})
                    continue
                video = torch.cat(videos, dim=0).to(DEVICE, dtype=next(MODEL.parameters()).dtype)
                features = MODEL(video)
                if len(videos) > 1:
                    features = features.reshape(len(videos), -1, features.shape[-1]).flatten(0, 1).unsqueeze(0)
                logits = CLASSIFIER(features)
                probs = F.softmax(logits.float(), dim=-1)[0]
                k = max(1, min(int(request.get("topk", 5)), probs.numel()))
                values, indices = probs.topk(k)
            predictions = [{"index": int(i), "label": LABELS.get(str(int(i)), str(int(i))),
                            "probability": float(v)} for v, i in zip(values.cpu(), indices.cpu())]
            await ws.send_json({"predictions": predictions, "feature_shape": list(features.shape)})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/VJEPA_checkpoint/vitg-384.pt")
    parser.add_argument("--classifier", default="/data/VJEPA_checkpoint/ssv2-vitg-384-64x2x3.pt")
    parser.add_argument("--labels", default="/data/VJEPA_checkpoint/ssv2_classes.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    load_models(args.model, args.classifier, args.labels, args.device, args.frames, args.size)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
