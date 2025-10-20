# Command - python ppt_to_gif_2.py "D:\Windows Folders\Desktop\Hubble_GIF_Input.pptx" "D:\Windows Folders\Desktop\Hubble\static\images\Slide_GIFs" 9,10,11,12,13 0.05 10

import os
import sys
import time
import win32com.client
from moviepy.editor import (
    VideoFileClip,
    vfx,
    concatenate_videoclips,
    ImageClip,  # <-- Added ImageClip import
)


def add_extra_pause(slide_clip, extra_pause):
    """
    Appends a static pause of 'extra_pause' seconds using the last frame.
    """
    # Get the image data of the very last frame
    # Use a small offset before the end to safely get the frame
    last_frame_time = slide_clip.duration - 0.01
    if last_frame_time < 0:
        last_frame_time = 0  # Handle very short clips

    last_frame_image = slide_clip.get_frame(last_frame_time)

    # Create a new clip from that single frame, and set its duration
    pause_clip = ImageClip(last_frame_image).set_duration(extra_pause)
    # Set the fps for the new clip to match the original, or a default
    pause_clip.fps = slide_clip.fps if slide_clip.fps else 10

    # Join the original clip with the new pause clip
    return concatenate_videoclips([slide_clip, pause_clip])


def ppt_to_gif(
    ppt_path,
    output_folder,
    slide_nums=None,
    speed_factor=1.0,
    extra_pause=0,
    fps=10,
):
    os.makedirs(output_folder, exist_ok=True)

    powerpoint = None
    presentation = None
    try:
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        powerpoint.Visible = 1
        # Use os.path.abspath to ensure COM object gets a full path
        full_ppt_path = os.path.abspath(ppt_path)
        presentation = powerpoint.Presentations.Open(full_ppt_path, WithWindow=False)

        num_slides = presentation.Slides.Count
        print(f"Found {num_slides} slides.")

        # Validate slide numbers
        if slide_nums:
            slide_nums = [s for s in slide_nums if 1 <= s <= num_slides]
        else:
            slide_nums = list(range(1, num_slides + 1))

        # Export full presentation to video
        video_path = os.path.join(output_folder, "presentation.mp4")
        print("Exporting presentation to video (animations preserved)...")
        # Use a default duration for slides without timings (e.g., 5 seconds)
        presentation.CreateVideo(
            video_path, UseTimingsAndNarrations=True, DefaultSlideDuration=5
        )

        # === NEW: Robust wait for video creation ===
        print("Waiting for PowerPoint to finish rendering video...")
        # 1 = ppCreateVideoStatusInProgress
        while presentation.CreateVideoStatus == 1:
            time.sleep(1)  # Poll every second

        if presentation.CreateVideoStatus == 2:  # 2 = ppCreateVideoStatusFailed
            raise Exception("PowerPoint video export failed.")

        print("Video rendering complete.")
        # Brief pause to ensure file handle is released by PowerPoint
        time.sleep(3)
        # === End of new wait logic ===

        clip = VideoFileClip(video_path)
        total_duration = clip.duration
        per_slide_duration = total_duration / num_slides
        # Increased epsilon slightly to better avoid transitions
        epsilon = 0.1

        print(
            f"Total video duration: {total_duration:.2f}s, "
            f"approx {per_slide_duration:.2f}s per slide"
        )

        for i in slide_nums:
            start = (i - 1) * per_slide_duration + epsilon
            # === FIXED: Subtract epsilon from end time to avoid next slide ===
            end = min(i * per_slide_duration - epsilon, total_duration)

            # Check if the calculated time is valid
            if start >= end:
                print(
                    f"Skipping slide {i} (calculated duration {end-start:.2f}s is negative or zero)."
                )
                continue

            print(f"Creating GIF for slide {i} ({start:.2f}-{end:.2f}s)...")
            slide_clip = clip.subclip(start, end)

            # Apply speed factor
            if speed_factor != 1.0:
                slide_clip = slide_clip.fx(vfx.speedx, speed_factor)

            # Apply extra pause at the end
            if extra_pause > 0:
                # === FIXED: Call to new function signature ===
                slide_clip = add_extra_pause(slide_clip, extra_pause)

            slide_gif_path = os.path.join(output_folder, f"slide_{i}.gif")
            slide_clip.write_gif(slide_gif_path, fps=fps)
            slide_clip.close()
            print(f"Saved: {slide_gif_path}")

        clip.close()

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        # Ensure PowerPoint objects are closed and quit
        if presentation:
            presentation.Close()
        if powerpoint:
            powerpoint.Quit()

    print("✅ Selected slides converted to GIFs successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python ppt_to_gif.py <path_to_pptx> <output_folder> "
            "[slide_nums_comma_separated] [speed_factor] [extra_pause_seconds]"
        )
        sys.exit(1)

    ppt_path = sys.argv[1]
    output_folder = sys.argv[2]

    slide_nums = None
    speed_factor = 1.0
    extra_pause = 0
    fps = 10

    args = sys.argv[3:]
    if args:
        # First argument could be comma-separated slide numbers
        try:
            slide_nums = [int(s.strip()) for s in args[0].split(",")]
            args = args[1:]  # remaining args
        except ValueError:
            # Not a list of numbers, assume it's speed_factor
            slide_nums = None

    # Remaining args: speed_factor and extra_pause
    if len(args) >= 1:
        try:
            speed_factor = float(args[0])
        except ValueError:
            print(f"Warning: Invalid speed_factor '{args[0]}', using 1.0")
            speed_factor = 1.0
    if len(args) >= 2:
        try:
            extra_pause = float(args[1])
        except ValueError:
            print(f"Warning: Invalid extra_pause '{args[1]}', using 0.")
            extra_pause = 0

    if not os.path.exists(ppt_path):
        print("❌ PowerPoint file not found.")
        sys.exit(1)

    ppt_to_gif(
        ppt_path, output_folder, slide_nums, speed_factor, extra_pause, fps=fps
    )