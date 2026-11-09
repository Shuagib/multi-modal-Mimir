import os
import argparse, io, json, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError
from vision_model.model import SigLIP2
from typing import NamedTuple
from collections.abc import Callable


### LOGGING
import transformers
transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.WARNING) # print info/warning/error
log = logging.getLogger("extract")
log.setLevel(logging.INFO)

for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# python -m pipeline.training.extract_vm_features --out data --n 100 --eval-n 10 --shard-size 5 --batch-size 4 --save-eval-images

# pipeline/training/sources.py

class Source(NamedTuple):
    """How to turn one HF row in to (image, prompt, target)."""
    hf_path: str
    split: str
    image_field: str
    to_text: Callable[[dict], tuple[str, str]]

def _cap(row:dict) -> tuple[str, str]:
    return "Describe the image", row["caption"]

def _count(row: dict) -> tuple[str, str]:
    return f"Hor many {row['label']} are there in the image", str(row("count"))

def _cap_qa(row: dict) -> tuple[str, str]:
    return f"Hor many {row['label']} are there in the image", str(row("count"))

SOURCES = {
    "pixmo-cap":    Source("allenai/pixmo-cap",    "tr" \
    "ain", "image_url", _cap),
    "pixmo-count":  Source("allenai/pixmo-count",  "train", "image_url", _count),
    "pixmo-cap-qa": Source("allenai/pixmo-cap-qa", "train", "image_url", _cap_qa),
}

# feature extraction pipeline

TIMEOUT = 10
HEADERS = {"User-Agent": "multi-modal-Mimir/0.1"}

def fetch(url: str) -> Image.Image | None:
    """Fetch image urls"""
    try:
        r = requests.get(url, timeout=TIMEOUT, headers = HEADERS)
        r.raise_for_status() # status for url
        return Image.open(io.BytesIO(r.content)).convert("RGB") # return image
    except (requests.RequestException, UnidentifiedImageError, OSError) as e:
        log.debug("skip %s: %s", url, e)
        # log.debug(f"skip {url}: {e}") #same but 0 cost for try
        return None

class Sample(NamedTuple):
    image: Image.Image
    caption: str
    url: str

class FeatureSample(NamedTuple):
    image: np.ndarray
    caption: str
    url: str

class ShardWriter:
    """Collects samples and writes them to disk in fixed-size batches."""

    def __init__(self, out_dir: Path, shard_size: int, pipeline_mode: str):
        self.out_dir = Path(out_dir)
        self.shard_size = shard_size
        self.pipeline_mode = pipeline_mode
        self.out_dir.mkdir(parents=True,exist_ok=True)
        self.buffer: list[FeatureSample] = []
        self.current_shard_id = 0
        self.total_added_samples = 0

    def add(self, feature: np.ndarray, caption: str, url: str) -> None:
        self.buffer.append(FeatureSample(feature, caption, url))
        self.total_added_samples += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Writes shards of data to the disk"""
        if not self.buffer:
            return
        feature_stem = f"{self.current_shard_id:05d}" # like f"{1234:05d}" ->  "01234" from features-01234.npy
        np.save(self.out_dir / f"features-{feature_stem}.npy",
            np.stack([s.image for s in self.buffer]))
        with open(self.out_dir / f"caption-{feature_stem}.jsonl", "w") as fh:
            for s in self.buffer:
                fh.write(json.dumps({"caption": s.caption, "image_url": s.url}) + "\n")

        log.info("%s shard %s: %d samples", self.pipeline_mode, feature_stem, len(self.buffer))
        self.buffer = []
        self.current_shard_id += 1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True) # directory the shards get written to
    ap.add_argument("--n", type=int, default=100_000) # how many samples to keep before stopping
    ap.add_argument("--shard-size", type=int, default=2000) # samples per output file
    ap.add_argument("--batch-size", type=int, default=64) # images per SigLIP forward pass (GPU)
    ap.add_argument("--workers", type=int, default=32) # concurrent HTTP downloads (network)
    ap.add_argument("--model", default="google/siglip2-base-patch16-256") # encoder model
    ap.add_argument("--eval-n", type=int, default=2000) # number of eval images
    ap.add_argument("--save-eval-images", action="store_true") # save the images for eval
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    vm = SigLIP2(model_name=args.model, device=device)
    processor = vm.processor

    from datasets import load_dataset
    stream = load_dataset("allenai/pixmo-cap", split="train", streaming=True)
    
    kept, seen = 0, 0

    train_writer = ShardWriter(out_dir=args.out / "train", shard_size=args.shard_size, pipeline_mode="train")
    eval_writer = ShardWriter(out_dir=args.out / "eval", shard_size=args.shard_size, pipeline_mode="eval")

    images_dir = args.out / "eval" / "images"

    if args.save_eval_images:   
        images_dir.mkdir(parents=True, exist_ok=True)
        

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batch_rows: list[dict] = []
        for row in stream:
            seen += 1
            batch_rows.append(row)
            if len(batch_rows) < args.batch_size:
                continue

            images = list(pool.map(lambda r: fetch(r["image_url"]), batch_rows)) # collect images
            samples = [Sample(im, r["caption"], r["image_url"]) for im, r in zip(images, batch_rows) if im is not None] #add image-caption pair
            
            batch_rows = []
            if not samples:
                continue

            px = processor(images=[s.image for s in samples], return_tensors="pt").to(device)
            with torch.no_grad():
                output = vm.model(**px).last_hidden_state
            output = output.cpu().numpy().astype(np.float16)

            for feature, s in zip(output, samples):
                writer = eval_writer if kept < args.eval_n else train_writer
                if writer is eval_writer and args.save_eval_images:
                    s.image.save(images_dir / f"{kept:05d}.jpg", "JPEG", quality=92)
                
                writer.add(feature, s.caption, s.url) # append to buffer and flush if full
                kept += 1

            if kept >= args.n:
                break
            if seen % 2000 == 0:
                log.info("seen %d, kept %d (%.0f%% ok)", seen, kept, 100 * kept / seen)

    train_writer.flush()
    eval_writer.flush()
    log.info("done: %d kept of %d seen", kept, seen)

if __name__ == "__main__":
    main()
    os._exit(0)