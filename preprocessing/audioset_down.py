import os
import subprocess
import csv

def download_and_trim_video(video_id, start_time, end_time, output_dir):
    """
    Downloads a YouTube video and trims it to the given start and end time.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_video_path = os.path.join(output_dir,'video', f"{video_id}.mp4")
    output_video_path = os.path.join(output_dir,'video', f"{video_id}_{start_time}_{end_time}.mp4")
    output_audio_path = os.path.join(output_dir,'audio', f"{video_id}_{start_time}_{end_time}.wav")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(output_video_path):
        print("exist")
    else:

        try:
            # Step 1: Download the full YouTube video using yt-dlp
            subprocess.run(
                ["yt-dlp", url, "-o", temp_video_path, "-f", "mp4"],
                check=True
            )

            # Step 2: Trim the video using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", temp_video_path, "-ss", str(start_time),
                "-to", str(end_time), "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
                output_video_path
            ], check=True)

            
            # Step 3: Extract the audio as a WAV file
            subprocess.run([
                "ffmpeg", "-i", output_video_path, "-q:a", "0", "-map", "a", "-ar", "16000",
                output_audio_path
            ], check=True)

            print(f"Saved video: {output_video_path}")
            print(f"Saved audio: {output_audio_path}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing video {video_id}: {e}")
        finally:
            # Clean up temporary video file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


def download_and_extract_audio(video_id, start_time, end_time, output_dir, browser="chrome"):
    """
    Downloads a YouTube video and extracts the audio as WAV within the given start and end time.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_audio_path = os.path.join(output_dir, 'audio', f"{video_id}_temp.wav")
    output_audio_path = os.path.join(output_dir, 'audio', f"{video_id}_{start_time}_{end_time}.wav")
    os.makedirs(os.path.join(output_dir, 'audio'), exist_ok=True)
    if os.path.isfile(output_audio_path):
        print("Audio already exists:", output_audio_path)
        return

    try:
        # Step 1: Download the full YouTube video using yt-dlp with cookies
        temp_video_path = os.path.join(output_dir, 'temp_video.mp4')
        subprocess.run(
            [
                "yt-dlp", url, "-o", temp_video_path, "-f", "bestaudio",
                "--cookies-from-browser", browser  # Automatically use browser cookies
            ],
            check=True
        )

        # Step 2: Extract audio as WAV from the downloaded video
        subprocess.run([
            "ffmpeg", "-i", temp_video_path, "-ss", str(start_time), "-to", str(end_time),
            "-q:a", "0", "-map", "a", "-ar", "16000", temp_audio_path
        ], check=True)

        # Step 3: Rename the file to final output path
        os.rename(temp_audio_path, output_audio_path)

        print(f"Saved audio: {output_audio_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error processing video {video_id}: {e}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def process_audioset_csv(csv_file, output_dir):
    """
    Reads AudioSet CSV metadata and processes each video.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            if row[0].split(' ')[0] != '#':
                video_id = row[0]
                start_time = float(row[1])
                end_time = float(row[2])

                # Process video
                if end_time > start_time:  # Ensure valid duration
                    download_and_trim_video(video_id, start_time, end_time, output_dir)


def process_audiocap_csv(csv_file, output_dir):
    import pandas as pd
    """
    Reads AudioSet CSV metadata and processes each video.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    for i in range(len(df)):
        video_id = df['youtube_id'].iloc[i]
        start_time = float(df['start_time'].iloc[i])
        end_time = start_time + 10.0

        download_and_trim_video(video_id, start_time, end_time, output_dir)


csv_file = "./dataset/audio/audioset/train.csv"  
output_dir = "./dataset/audio/audioset/train"
process_audiocap_csv(csv_file, output_dir)
