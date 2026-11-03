import argparse, io, json, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError
from vision_model.model import SigLIP2

logging.basicConfig(level=logging.INFO) # print info/warning/error
log = logging.getLogger("extract")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", default="google/siglip2-base-patch16-256")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    vm = SigLIP2(model_name=args.model, device=device)
    processor = vm.processor

    from datasets import load_dataset
    stream = load_dataset("allenai/pixmo-cap", split="train", streaming=True)

    shard, shard_id, kept, seen = [], 0, 0, 0

    def flush() -> None:
        """Writes shards of data to the disk"""
        nonlocal shard, shard_id
        if not shard:
            return
        features = np.stack([f for f, _ in shard])
        np.save(args.out / f"features-{shard_id:05d}.npy", features)
    
        with open(args.out / f"features-{shard_id:05d}.jsonl", "w") as fh:
            for _, cap in shard:
                fh.write(json.dumps({"caption": cap}))

        log.info("shard %d: %d samples", shard_id, len(shard))
        shard, shard_id = [], shard_id + 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batch_rows: list[dict] = []
        for row in stream:
            seen += 1
            batch_rows.append(row)
            if len(batch_rows) < args.batch_size:
                continue

            images = list(pool.map(lambda r: fetch(r["images_url"]), batch_rows)) # collect images
            pairs = [(im, r["caption"]) for im, r in zip(images, batch_rows) if im] #add image-caption pair
            batch_rows = []
            if not pairs:
                continue

            px = processor(images=[im for im, _ in pairs], return_tensors="pt")
            with torch.no_grad():
                output = vm(**px).last_hidden_state
            output = output.cpu().numpy().astype(np.float16)

            for feature, (_,caption) in zip(output, pairs):
                shard.append(feature, caption)
                kept += 1
                if len(shard) >= args.shard_size:
                    flush()

            if kept >= args.n:
                break
            if seen % 2000 == 0:
                log.info("seen %d, kept %d (%.0f%% ok)", seen, kept, 100 * kept / seen)

    flush()
    log.info("done: %d kept of %d seen", kept, seen)

if __name__ == "__main__":
    main()

