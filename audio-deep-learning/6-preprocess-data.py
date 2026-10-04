import json
import math
from pathlib import Path

import kagglehub
import librosa
from tqdm import tqdm
from dotenv import load_dotenv

# constants
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(
    dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5
):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length
    )

    genre_dirs = sorted(p for p in dataset_path.iterdir() if p.is_dir())

    # Outer progress bar: tracks overall progress across all genres
    genre_pbar = tqdm(genre_dirs, desc="Total Genres", unit="genre")

    for i, genre_dir in enumerate(genre_pbar):
        semantic_label = genre_dir.name
        data["mapping"].append(semantic_label)

        wav_files = sorted(genre_dir.glob("*.wav"))

        # Inner progress bar: tracks files within the current genre
        # leave=False removes the inner bar once that genre finishes
        file_pbar = tqdm(
            wav_files,
            desc=f"{semantic_label:<10}",
            leave=False,
            unit="track",
        )

        for file_path in file_pbar:
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                # Use tqdm.write instead of print to prevent breaking the bar display
                tqdm.write(f"Skipping corrupted file {file_path.name}: {e}")
                continue

            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                end_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(
                    y=signal[start_sample:end_sample],
                    sr=sr,
                    n_fft=n_fft,
                    n_mfcc=n_mfcc,
                    hop_length=hop_length,
                ).T

                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i)

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    tqdm.write("\nData Processing Finished!")
    tqdm.write("=====================")

if __name__ == "__main__":
    load_dotenv()
    download_path = Path(
        kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
    )
    print(f"download path: {download_path}")
    print(f"data contents: {list(download_path.iterdir())}")

    json_path = Path("../data/audio-deep-learning/mfcc_data.json")
    dataset_path = download_path / "Data" / "genres_original"
    save_mfcc(dataset_path, json_path, num_segments=10)
