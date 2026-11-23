import os
import argparse, io, json, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datasets import load_dataset
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
    return f"How many {row['label']} are there in the image", str(row["count"])

def _cap_qa(row: dict) -> tuple[str, str]:
    question = row["question"].replace("[USER]", "").replace("[ASSISTANT]", "").strip()
    return question, row["answer"]

SOURCES = {
    "pixmo-cap":    Source("allenai/pixmo-cap",    "train", "image_url", _cap),
    "pixmo-count":  Source("allenai/pixmo-count",  "train", "image_url", _count),
    "pixmo-cap-qa": Source("allenai/pixmo-cap-qa", "train", "image_url", _cap_qa),
}

# approximate rows per source (row count minus some failed downloads)
CAPACITY = {
    "pixmo-cap":    500_000,
    "pixmo-count":   15_000,
    "pixmo-cap-qa": 150_000,
}

def quotas(total: int, capacities: dict[str, int]) -> dict[str, int]:
        """Split total evenly; sources that run out give their remainder to the others."""
        q, left, open_ = {}, total, dict(capacities)
        while open_ and left > 0:
            share = left // len(open_)
            small = {k: v for k, v in open_.items() if v <= share}
            if not small:
                for k in open_: q[k] = share
                break
            for k, v in small.items():
                q[k] = v
                left -= v
                del open_[k]
        return q

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
    prompt: str
    caption: str
    url: str

class FeatureSample(NamedTuple):
    image: np.ndarray
    prompt: str
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

    def add(self, feature: np.ndarray, prompt: str, caption: str, url: str) -> None:
        self.buffer.append(FeatureSample(feature, prompt, caption, url))
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
            fh.writelines(json.dumps({"prompt": s.prompt, "caption": s.caption, "image_url": s.url}) + "\n" for s in self.buffer)

        log.info("%s shard %s: %d samples", self.pipeline_mode, feature_stem, len(self.buffer))
        self.buffer = []
        self.current_shard_id += 1

def extract_source(name, src, args, vm, device, pool, train_writer, eval_writer, test_writer, images_dir, train_q, eval_q, test_q):
    train_n=train_q[name]
    eval_n=eval_q[name]
    test_n = test_q[name]

    stream = load_dataset(src.hf_path, split=src.split, streaming=True)
    kept, seen, batch_rows = 0, 0, []

    for row in stream:
        seen += 1
        batch_rows.append(row)
        if len(batch_rows) < args.batch_size:
            continue

        images = list(pool.map(lambda r: fetch(r["image_url"]), batch_rows)) # collect images
        samples = [Sample(im, *src.to_text(r), r["image_url"]) for im, r in zip(images, batch_rows) if im is not None] #add image-caption pair
        
        batch_rows = []
        if not samples:
            continue

        px = vm.processor(images=[s.image for s in samples], return_tensors="pt").to(device)
        with torch.no_grad():
            output = vm.model(**px).last_hidden_state
        output = output.cpu().numpy().astype(np.float16)

        for feature, s in zip(output, samples):
            if kept < eval_n:
                writer = eval_writer
            elif kept < eval_n + test_n:
                writer = test_writer
            else:
                writer = train_writer
            if writer is eval_writer and args.save_eval_images:
                s.image.save(images_dir / f"{name}-{kept:05d}.jpg", "JPEG", quality=92)
            
            writer.add(feature, s.prompt, s.caption, s.url) # append to buffer and flush if full
            kept += 1
            if kept >= eval_n + test_n + train_n:
                break
        if kept >= eval_n + test_n + train_n:
            break
        log.info("seen %d, kept %d (%.0f%% ok)", seen, kept, 100 * kept / seen)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True) # directory the shards get written to
    ap.add_argument("--n", type=int, default=100_000) # how many samples to keep before stopping
    ap.add_argument("--shard-size", type=int, default=2000) # samples per output file
    ap.add_argument("--batch-size", type=int, default=64) # images per SigLIP forward pass (GPU)
    ap.add_argument("--workers", type=int, default=32) # concurrent HTTP downloads (network)
    ap.add_argument("--model", default="google/siglip2-base-patch16-256") # encoder model
    ap.add_argument("--eval-n", type=int, default=2000) # number of eval images
    ap.add_argument("--test-n", type=int, default=100) # small held-out test set
    ap.add_argument("--save-eval-images", action="store_true") # save the images for eval
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))

    args = ap.parse_args()

    caps = {name: CAPACITY[name] for name in args.sources}
    train_q = quotas(args.n, caps)
    eval_q = quotas(args.eval_n, caps)
    test_q = quotas(args.test_n, caps)

    log.info("train quotas: %s | eval quotas: %s | test quotas: %s", train_q, eval_q, test_q)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    vm = SigLIP2(model_name=args.model, device=device)    

    train_writer = ShardWriter(out_dir=args.out / "train", shard_size=args.shard_size, pipeline_mode="train")
    eval_writer = ShardWriter(out_dir=args.out / "eval", shard_size=args.shard_size, pipeline_mode="eval")
    test_writer = ShardWriter(out_dir=args.out / "test", shard_size=args.shard_size, pipeline_mode="test")

    images_dir = args.out / "eval" / "images"

    if args.save_eval_images:   
        images_dir.mkdir(parents=True, exist_ok=True)
        
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in args.sources:
            extract_source(
                name=name, 
                src=SOURCES[name], 
                args=args, 
                vm=vm, 
                device=device, 
                pool=pool,
                train_writer=train_writer,
                eval_writer=eval_writer,
                test_writer=test_writer,
                images_dir=images_dir,
                train_q=train_q,
                eval_q=eval_q,
                test_q=test_q
                )

    train_writer.flush()
    eval_writer.flush()
    test_writer.flush()

if __name__ == "__main__":
    main()
    os._exit(0)