import argparse
import os
import cv2  # Make sure to import OpenCV for live feed
import numpy as np  # To handle the frames
from vitallens import VitalLens, Method

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run VitalLens for vital sign estimation from video.")
    parser.add_argument('--method', type=str, default='VITALLENS', choices=['VITALLENS', 'POS', 'CHROM', 'G'],
                        help='Inference method to use.')
    parser.add_argument('--video_path', type=str, help='Path to the video file.')
    parser.add_argument('--livefeed', action='store_true', help='Use live feed from laptop front camera.')
    parser.add_argument('--api_key', type=str, required=False, help='API key for VitalLens API.')
    parser.add_argument('--detect_faces', type=bool, default=True, help='Whether to detect faces in the video.')
    parser.add_argument('--estimate_running_vitals', type=bool, default=True, help='Whether to estimate running vitals.')
    parser.add_argument('--export_to_json', type=bool, default=True, help='Whether to export results to a JSON file.')
    parser.add_argument('--export_dir', type=str, default='.', help='Directory to export JSON files.')
    #parser.add_argument('--csv_filename', type=str, default='vital_data.csv', help='Filename for the CSV export.')

    return parser.parse_args()
def capture_live_feed(num_frames=900):
    """Capture live feed from the laptop front camera for a specified number of frames."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        # Display the current frame in a window
        cv2.imshow("Live Feed", frame) 
        frames.append(frame)
        # Close the window if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Get the frame rate of the webcam feed
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    cv2.destroyAllWindows()

    return frames, fps

def main():
    args = parse_arguments()
    # Live feed input
    if args.livefeed:
        # Capture live feed or load video based on the argument
        print("Capturing live feed from webcam...")
        frames, fps = capture_live_feed()
        print(f"Captured {len(frames)} frames from live feed at {fps:.2f} FPS.")

        # Process the live feed frames with VitalLens
        method = getattr(Method, args.method)
        vl = VitalLens(
                method=method,
                api_key=args.api_key,
                detect_faces=args.detect_faces,
                estimate_running_vitals=args.estimate_running_vitals,
                export_to_json=args.export_to_json,
                export_dir=args.export_dir)

        result = vl(frames)  # Pass the frames directly to the VitalLens instance

    # Video file input
    else:
        # Ensure the video file exists
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file {args.video_path} does not exist.")

        # Set up the VitalLens instance
        method = getattr(Method, args.method)
        vl = VitalLens(
            method=method,
            api_key=args.api_key,
            detect_faces=args.detect_faces,
            estimate_running_vitals=args.estimate_running_vitals,
            export_to_json=args.export_to_json,
            export_dir=args.export_dir
        )

        # Run the estimation on the video file
        result = vl(args.video_path)
    # Print the results and save data to CSV
    for face_result in result:
        print("Result: ")
        print("Face coordinates:", face_result['face']['coordinates'])
        print("Heart rate:", face_result['vital_signs']['heart_rate'])
        print("PPG waveform:", face_result['vital_signs']['ppg_waveform'])
        print("Message:", face_result['message'])


if __name__ == "__main__":
    main()

