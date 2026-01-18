import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor


def _as_numpy_mask(mask: torch.Tensor) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        if mask.dtype.is_floating_point:
            mask = mask > 0.5
        mask = mask.to(torch.uint8) * 255
        return mask.numpy()
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def _as_list(tensor_or_list) -> list:
    if torch.is_tensor(tensor_or_list):
        return tensor_or_list.detach().cpu().tolist()
    return list(tensor_or_list)


def _create_run_dir(image_path: str, out_dir: str) -> Path:
    base = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{base}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _draw_boxes(image: Image.Image, boxes: list, scores: list) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for idx, box in enumerate(boxes):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if idx < len(scores):
            label = f"{scores[idx]:.3f}"
            draw.text((x1 + 4, y1 + 4), label, fill="red")
    return annotated


def run_image(image_path: str, prompt: str, out_dir: str) -> None:
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    run_dir = _create_run_dir(image_path, out_dir)
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    boxes_list = _as_list(boxes)
    scores_list = _as_list(scores)

    image.convert("RGB").save(run_dir / "original.png")
    annotated = _draw_boxes(image, boxes_list, scores_list)
    annotated.save(run_dir / "annotated_boxes.png")

    for idx, mask in enumerate(masks):
        mask_array = _as_numpy_mask(mask)
        Image.fromarray(mask_array).save(masks_dir / f"mask_{idx:03d}.png")

    with open(run_dir / "detections.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "image": str(Path(image_path).name),
                "prompt": prompt,
                "boxes": boxes_list,
                "scores": scores_list,
            },
            handle,
            indent=2,
        )

    print(f"Saved outputs to: {run_dir}")


def run_video(video_path: str, prompt: str) -> None:
    video_predictor = build_sam3_video_predictor()
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=response["session_id"],
            frame_index=0,
            text=prompt,
        )
    )
    _ = response["outputs"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--prompt", default="object", help="Text prompt for SAM3.")
    parser.add_argument(
        "--out-dir",
        default="sam3_outputs",
        help="Base output directory for saved results.",
    )
    parser.add_argument(
        "--video",
        help="Optional video path (JPEG folder or MP4) to run video mode.",
    )
    args = parser.parse_args()

    run_image(args.image, args.prompt, args.out_dir)
    if args.video:
        run_video(args.video, args.prompt)


if __name__ == "__main__":
    main()