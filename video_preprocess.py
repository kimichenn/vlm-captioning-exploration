import cv2
import os


def extract_frames(video_path, output_folder="frames"):
    """
    Extracts all frames from a video and saves them as PNG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where extracted frames will be stored.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        # Break the loop if no frame is returned
        if not ret:
            break

        # Construct the output file path
        frame_filename = os.path.join(
            output_folder, f"frame_{frame_count:06d}.png"
        )

        # Save the current frame as a PNG image
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture
    cap.release()
    print(
        f"Extraction complete. {frame_count} frames saved in '{output_folder}' folder."
    )


if __name__ == "__main__":
    video_path = "output_test_smb/output_video_1.mp4"
    extract_frames(video_path, output_folder="results/test_set_new")
