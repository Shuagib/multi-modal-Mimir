import hashlib
import io
import logging
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "multi-modal-Mimir/0.1 (research)"


class ImageCaptionDataset(Dataset):
    """Image-caption pairs where images are referenced by URL and cached on disk.

    pixmo-cap ships captions plus `image_url`; the images themselves are not
    included and a meaningful share of the URLs have rotted since publication, so
    fetch failures are expected and are reported as `None` for the caller to skip.
    """

    def __init__(
        self,
        records: list[dict],
        cache_dir: Path,
        url_field: str = "image_url",
        caption_field: str = "caption",
    ) -> None:
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url_field = url_field
        self.caption_field = caption_field

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.img"

    def _load_image(self, url: str) -> Image.Image | None:
        path = self._cache_path(url)
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                logger.warning("corrupt cache entry, refetching: %s", path)
                path.unlink(missing_ok=True)
        try:
            response = requests.get(
                url, timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            logger.warning("failed to fetch %s: %s", url, exc)
            return None
        path.write_bytes(response.content)
        return image

    def __getitem__(self, index: int) -> dict | None:
        record = self.records[index]
        url = record.get(self.url_field)
        caption = record.get(self.caption_field)
        if not url or not caption:
            return None
        image = self._load_image(url)
        if image is None:
            return None
        return {"image": image, "caption": caption}


def load_pixmo_cap_records(
    dataset_name: str, split: str, max_samples: int | None
) -> list[dict]:
    """Pull caption metadata only; images are fetched lazily per item."""
    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True)
    records: list[dict] = []
    for row in stream:
        records.append({"image_url": row["image_url"], "caption": row["caption"]})
        if max_samples is not None and len(records) >= max_samples:
            break
    return records
