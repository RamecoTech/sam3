import argparse
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor


def run_image(image_path: str, prompt: str) -> None:
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"masks: {len(masks)}, boxes: {len(boxes)}, scores: {len(scores)}")


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
        "--video",
        help="Optional video path (JPEG folder or MP4) to run video mode.",
    )
    args = parser.parse_args()

    run_image(args.image, args.prompt)
    if args.video:
        run_video(args.video, args.prompt)


if __name__ == "__main__":
    main()