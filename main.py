"""
main.py

Simple text-to-video generator with:
- subject + action prompt
- style preset
- camera angle preset
- approximate duration control
- extra feature: "prompt strictness" (guidance scale)

Output: MP4 video file saved to disk.
"""

import os
import textwrap

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


# ----------------------------
# 1. Global style + camera maps
# ----------------------------

STYLE_PRESETS = {
    "cartoon": "in a colorful 2D cartoon animation style, flat shading, bold outlines",
    "cinematic": "cinematic film lighting, realistic shading, depth of field",
    "3d": "in a 3D Pixar-like animation style, soft lighting, detailed textures",
    "sketch": "in a hand-drawn pencil sketch style, black and white, cross-hatching",
}

CAMERA_PRESETS = {
    "top-down": "top-down view, bird’s-eye angle",
    "side": "side view shot at eye level",
    "close-up": "close-up shot, showing only the subject’s upper body",
    "wide": "wide-angle establishing shot, full scene visible",
}


# ----------------------------
# 2. Model loading
# ----------------------------

def load_text_to_video_model():
    """
    Loads the text-to-video diffusion pipeline.

    Uses a publicly available text-to-video model from Hugging Face diffusers.
    """
    model_id = "damo-vilab/text-to-video-ms-1.7b"  # you can change this if needed

    print(f"[INFO] Loading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    if torch.cuda.is_available():
        device = "cuda"
        pipe = pipe.to(device)
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        print("[INFO] Using CUDA GPU")
    else:
        device = "cpu"
        pipe = pipe.to(device)
        print("[WARN] CUDA GPU not available. Generation may be VERY slow on CPU.")

    return pipe


# ----------------------------
# 3. Input handling
# ----------------------------

def get_user_inputs():
    """
    Ask the user for all required inputs via CLI.
    """
    print("=== Single-Video Generator ===")
    print("Fill in the fields below. Press Enter to accept defaults where shown.\n")

    base_prompt = input("Subject + action (e.g., 'a dog running across a field'): ").strip()
    if not base_prompt:
        base_prompt = "a cat dancing in a living room"
        print(f"[INFO] Empty input, using default: {base_prompt}")

    print("\nAvailable styles:", ", ".join(STYLE_PRESETS.keys()))
    style = input("Style [cartoon/cinematic/3d/sketch] (default: cartoon): ").strip().lower()
    if style not in STYLE_PRESETS:
        print("[WARN] Unknown style, defaulting to 'cartoon'")
        style = "cartoon"

    print("\nAvailable camera angles:", ", ".join(CAMERA_PRESETS.keys()))
    camera = input("Camera angle [top-down/side/close-up/wide] (default: side): ").strip().lower()
    if camera not in CAMERA_PRESETS:
        print("[WARN] Unknown camera angle, defaulting to 'side'")
        camera = "side"

    duration_str = input("\nDuration in seconds (1–10, default: 4): ").strip()
    try:
        duration = float(duration_str) if duration_str else 4.0
    except ValueError:
        print("[WARN] Invalid duration, defaulting to 4 seconds.")
        duration = 4.0
    duration = max(1.0, min(duration, 10.0))

    guidance_str = input(
        "\nPrompt strictness [1–20] "
        "(how strongly the video should follow the text, default: 12): "
    ).strip()
    try:
        guidance = float(guidance_str) if guidance_str else 12.0
    except ValueError:
        print("[WARN] Invalid value, defaulting to 12.")
        guidance = 12.0
    guidance = max(1.0, min(guidance, 20.0))

    return base_prompt, style, camera, duration, guidance


def build_prompt(base_prompt, style_key, camera_key):
    """
    Combine base prompt with style and camera descriptors.
    """
    style_desc = STYLE_PRESETS.get(style_key, "")
    cam_desc = CAMERA_PRESETS.get(camera_key, "")

    full_prompt = base_prompt
    if style_desc:
        full_prompt += f", {style_desc}"
    if cam_desc:
        full_prompt += f", {cam_desc}"

    print("\n[DEBUG] Final text prompt for the model:")
    print(textwrap.fill(full_prompt, width=80))
    return full_prompt


# ----------------------------
# 4. Video generation
# ----------------------------

def generate_video(
    pipe,
    prompt,
    duration_seconds=4.0,
    fps=8,
    num_inference_steps=25,
    guidance_scale=12.0,
    output_path="output.mp4",
):
    """
    Generates a short video using the text-to-video pipeline.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base_num_frames = 16
    approx_single_duration = base_num_frames / fps
    repeats = max(1, int(round(duration_seconds / approx_single_duration)))

    print(f"\n[INFO] Target duration: {duration_seconds:.1f}s at {fps} FPS")
    print(f"[INFO] Each generation ≈ {approx_single_duration:.2f}s")
    print(f"[INFO] Number of chunks to generate: {repeats}")

    all_frames = []

    for i in range(repeats):
        print(f"[INFO] Generating chunk {i + 1}/{repeats}...")
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        video_frames = result.frames[0]
        all_frames.extend(video_frames)

    print(f"[INFO] Total frames generated: {len(all_frames)}")

    video_path = export_to_video(
        all_frames,
        fps=fps,
        output_video_path=output_path,
    )
    print(f"[INFO] Saved video to: {video_path}")
    return video_path, len(all_frames), fps


# ----------------------------
# 5. Qualitative evaluation helper
# ----------------------------

def print_qualitative_evaluation_template(
    user_prompt,
    style,
    camera,
    duration,
    guidance,
    video_path,
    total_frames,
    fps,
):
    """
    Print a short template to guide your qualitative evaluation.
    """
    print("\n=== Qualitative Evaluation Notes ===")
    print(f"Video file: {video_path}")
    print(f"Frames: {total_frames} at {fps} FPS (≈ {total_frames / fps:.2f}s)")
    print(f"User prompt (subject + action): {user_prompt}")
    print(f"Style preset: {style}")
    print(f"Camera preset: {camera}")
    print(f"Prompt strictness (guidance scale): {guidance}")

    print(
        textwrap.dedent(
            """
            After watching the video, you can answer these in your report:

            1. Prompt adherence (subject + action):
               - Does the main subject match the text (e.g., cat, dog, person)?
               - Is the intended action visible (e.g., running, dancing, waving)?

            2. Style match:
               - Does the video look close to the chosen style (cartoon, cinematic, etc.)?
               - Do colors, shading, and textures match your expectation?

            3. Camera angle:
               - Does the framing resemble the requested angle (top-down, side, close-up, wide)?
               - Are perspective and composition roughly correct?

            4. Temporal consistency and motion:
               - Is the motion relatively smooth?
               - Are there any sudden jumps, flickers, or weird deformations?

            5. Effect of prompt strictness:
               - When you use a LOWER strictness value, does the video become more "free" or creative but less accurate?
               - When you use a HIGHER strictness value, does the video follow the text better but maybe look more noisy or distorted?

            You can compare 2–3 videos with different styles / camera angles / strictness
            and summarize your observations in the report.
            """
        )
    )


# ----------------------------
# 6. Main entry point
# ----------------------------

def main():
    pipe = load_text_to_video_model()

    base_prompt, style, camera, duration, guidance = get_user_inputs()

    full_prompt = build_prompt(base_prompt, style, camera)

    fps = 8
    output_path = "samples/generated_video.mp4"

    video_path, total_frames, fps = generate_video(
        pipe,
        full_prompt,
        duration_seconds=duration,
        fps=fps,
        num_inference_steps=25,
        guidance_scale=guidance,
        output_path=output_path,
    )

    print_qualitative_evaluation_template(
        user_prompt=base_prompt,
        style=style,
        camera=camera,
        duration=duration,
        guidance=guidance,
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
    )


if __name__ == "__main__":
    main()
