import os
import json
import random

import torch
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from torchvision import transforms


COCO_ROOT = "/mnt/sda/master/dataset/oymk/mscoco14/coco"
IMAGE_DIR = os.path.join(COCO_ROOT, "train2014")
CAPTION_FILE = os.path.join(
    COCO_ROOT,
    "annotations",
    "captions_train2014.json"
)

OUTPUT_DIR = "../data/mscoco"

N_SAMPLES = 10000
SEED = 42
BATCH_SIZE = 64


def load_pairs():
    with open(CAPTION_FILE, "r") as f:
        data = json.load(f)

    image_map = {
        item["id"]: item["file_name"]
        for item in data["images"]
    }

    captions = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]

        if image_id not in captions:
            captions[image_id] = ann["caption"]

    pairs = [
        (
            os.path.join(IMAGE_DIR, image_map[image_id]),
            captions[image_id],
        )
        for image_id in captions
        if image_id in image_map
    ]

    random.seed(SEED)
    random.shuffle(pairs)

    return pairs[:N_SAMPLES]


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    pairs = load_pairs()

    print("Number of pairs:", len(pairs))

    image_model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14"
    ).to(device)

    image_model.eval()

    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=str(device)
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_features = []
    text_features = []

    for start in tqdm(
        range(0, len(pairs), BATCH_SIZE)
    ):
        batch = pairs[start:start + BATCH_SIZE]

        images = [
            transform(
                Image.open(path).convert("RGB")
            )
            for path, _ in batch
        ]

        captions = [
            caption
            for _, caption in batch
        ]

        images = torch.stack(images).to(device)

        with torch.no_grad():

            img_feat = image_model(images)

            txt_feat = text_model.encode(
                captions,
                convert_to_tensor=True,
                device=str(device),
            )

            txt_feat = torch.nn.functional.pad(
                txt_feat,
                (0, 2048 - txt_feat.shape[1])
            )

        image_features.append(
            img_feat.cpu()
        )

        text_features.append(
            txt_feat.cpu()
        )

    image_features = torch.cat(
        image_features,
        dim=0
    )

    text_features = torch.cat(
        text_features,
        dim=0
    )

    print(
        "Image:",
        image_features.shape
    )

    print(
        "Text:",
        text_features.shape
    )

    torch.save(
        image_features,
        os.path.join(
            OUTPUT_DIR,
            "encoded1.pt"
        )
    )

    torch.save(
        text_features,
        os.path.join(
            OUTPUT_DIR,
            "encoded2.pt"
        )
    )


if __name__ == "__main__":
    main()