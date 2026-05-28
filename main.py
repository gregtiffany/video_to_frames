import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import cv2
import requests
from PIL import Image


SETTINGS_FILE = "settings.json"


def load_settings(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "onevizion" not in data or not isinstance(data["onevizion"], dict):
        raise ValueError("Missing 'onevizion' object in settings.json")

    if "processing" not in data or not isinstance(data["processing"], dict):
        raise ValueError("Missing 'processing' object in settings.json")

    ov = data["onevizion"]
    pr = data["processing"]

    required_ov = ["base_url", "access_key", "secret_key"]
    required_pr = [
        "trackor_type",
        "trigger_field",
        "source_video_field",
        "target_frames_field",
        "frame_interval_seconds",
        "max_image_width",
        "image_quality"
    ]

    missing_ov = [k for k in required_ov if k not in ov or ov[k] in (None, "")]
    missing_pr = [k for k in required_pr if k not in pr or pr[k] in (None, "")]

    if missing_ov:
        raise ValueError(f"Missing required onevizion setting(s): {', '.join(missing_ov)}")

    if missing_pr:
        raise ValueError(f"Missing required processing setting(s): {', '.join(missing_pr)}")

    pr.setdefault("working_dir", "work")

    interval = pr["frame_interval_seconds"]
    max_width = pr["max_image_width"]
    quality = pr["image_quality"]

    if not isinstance(interval, (int, float)) or interval <= 0:
        raise ValueError("frame_interval_seconds must be a number > 0")

    if not isinstance(max_width, int) or max_width <= 0:
        raise ValueError("max_image_width must be a positive integer")

    if not isinstance(quality, int) or quality < 1 or quality > 100:
        raise ValueError("image_quality must be an integer from 1 to 100")

    return data


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name)


def build_bearer_token(access_key: str, secret_key: str) -> str:
    # OneVizion token auth is documented as Bearer with "access_key:secret_key"
    return f"{access_key}:{secret_key}"


def create_session(access_key: str, secret_key: str) -> requests.Session:
    session = requests.Session()
    bearer_value = build_bearer_token(access_key, secret_key)
    session.headers.update({
        "Authorization": f"Bearer {bearer_value}"
    })
    return session


def is_triggered(value) -> bool:
    return value in (1, "1", True, "true", "TRUE", "True")


def get_trackor_id(trackor: dict):
    candidate_keys = [
        "trackorId",
        "trackor_id",
        "TRACKOR_ID",
        "id",
        "ID"
    ]
    for key in candidate_keys:
        if key in trackor and trackor[key] not in (None, ""):
            return trackor[key]
    return None


def normalize_trackor_list(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "items", "results", "trackors", "rows"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError("Could not determine trackor list from API response.")


def get_trackors(session: requests.Session, base_url: str, trackor_type: str, trigger_field: str):
    encoded_trackor_type = quote(str(trackor_type), safe="")
    encoded_trigger_field = quote(str(trigger_field), safe="")
    url = f"{base_url}/api/v3/trackor_types/{encoded_trackor_type}/trackors?{encoded_trigger_field}=1"

    print(f"Requesting triggered trackors from: {url}")
    response = session.get(url, headers={"Accept": "application/json"}, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to retrieve trackors. HTTP {response.status_code}: {response.text[:2000]}"
        )

    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list response, but got: {type(payload).__name__}")

    return payload


def guess_extension_from_response(response: requests.Response) -> str:
    content_type = (response.headers.get("Content-Type") or "").lower().split(";")[0].strip()
    mapping = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "application/zip": ".zip",
        "application/octet-stream": ""
    }
    return mapping.get(content_type, "")


def get_filename_from_headers(response: requests.Response):
    content_disp = response.headers.get("Content-Disposition", "")
    if not content_disp:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disp, re.IGNORECASE)
    if match:
        return safe_filename(match.group(1).strip())

    return None


def download_source_video(session: requests.Session, base_url: str, trackor_id, source_field: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    encoded_field_name = quote(str(source_field), safe="")
    url = f"{base_url}/api/v3/trackor/{trackor_id}/file/{encoded_field_name}"

    print(f"Downloading source video from: {url}")
    response = session.get(url, headers={"Accept": "*/*"}, stream=True, timeout=600)

    if response.status_code != 200:
        raise RuntimeError(f"Download failed for Trackor {trackor_id}. HTTP {response.status_code}: {response.text[:2000]}")

    header_filename = get_filename_from_headers(response)
    if header_filename:
        filename = header_filename
    else:
        ext = guess_extension_from_response(response)
        filename = f"trackor_{trackor_id}_{safe_filename(source_field)}{ext}"

    video_path = output_dir / filename

    with open(video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Source video field is expected to be a single E-File video, not a Multi E-File ZIP
    if video_path.suffix.lower() == ".zip":
        raise RuntimeError(
            f"Trackor {trackor_id}: source_video_field returned a ZIP archive. "
            "Use a single E-File video field for source_video_field."
        )

    print(f"Saved source video: {video_path}")
    return video_path


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    trackor_id,
    interval_seconds: float,
    max_width: int,
    jpeg_quality: int
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Could not determine FPS for video: {video_path}. "
            "FPS is required to capture frames at time intervals."
        )

    if not frame_count or frame_count <= 0:
        cap.release()
        raise RuntimeError(
            f"Could not determine frame count for video: {video_path}"
        )

    duration_seconds = frame_count / fps
    timestamps = [0.0]

    t = float(interval_seconds)
    while t < duration_seconds:
        timestamps.append(t)
        t += float(interval_seconds)

    saved_frames = []
    for idx, timestamp in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Warning: could not read frame at {timestamp:.2f}s for Trackor {trackor_id}")
            continue

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        if img.width > max_width:
            new_height = int(img.height * (max_width / img.width))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        frame_filename = f"trackor_{trackor_id}_frame_{idx:04d}_{int(round(timestamp))}s.jpg"
        frame_path = frames_dir / frame_filename

        img.save(frame_path, format="JPEG", quality=jpeg_quality, optimize=True)
        saved_frames.append(frame_path)

    cap.release()

    if not saved_frames:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    print(f"Extracted {len(saved_frames)} frame(s) from Trackor {trackor_id}")
    return saved_frames


def upload_file_to_multiefile(
    session: requests.Session,
    base_url: str,
    trackor_id,
    target_field: str,
    file_path: Path
):
    encoded_field_name = quote(str(target_field), safe="")
    url = f"{base_url}/api/v3/trackor/{trackor_id}/file/{encoded_field_name}"

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        files = {
            "file": (file_path.name, f, mime_type)
        }

        data = {
            "file_name": file_path.name
        }

        response = session.post(
            url,
            files=files,
            data=data,
            timeout=600
        )

    if response.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"Upload failed for Trackor {trackor_id}, file {file_path.name}. "
            f"HTTP {response.status_code}: {response.text[:2000]}"
        )

    print(f"Uploaded frame: {file_path.name}")


def reset_trigger_field(
    session: requests.Session,
    base_url: str,
    trackor_id,
    trigger_field: str,
    reset_value=0
):
    url = f"{base_url}/api/v3/trackors/{trackor_id}"

    # Different tenants/wrappers sometimes expect slightly different payload shapes.
    # We try a few safe variants.
    payload_candidates = [
        {trigger_field: reset_value},
        {"fields": {trigger_field: reset_value}},
        {"data": {trigger_field: reset_value}},
    ]

    last_error = None
    for payload in payload_candidates:
        response = session.put(
            url,
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=300
        )

        if response.status_code in (200, 201, 204):
            print(f"Reset {trigger_field} to {reset_value} for Trackor {trackor_id}")
            return

        last_error = f"HTTP {response.status_code}: {response.text[:2000]}"

    raise RuntimeError(
        f"Failed to reset trigger field for Trackor {trackor_id}. Last error: {last_error}"
    )


def process_trackor(
    session: requests.Session,
    base_url: str,
    trackor: dict,
    settings: dict
):
    pr = settings["processing"]

    trackor_id = get_trackor_id(trackor)
    if not trackor_id:
        raise RuntimeError("No trackor ID found in record.")

    trigger_field = pr["trigger_field"]
    source_video_field = pr["source_video_field"]
    target_frames_field = pr["target_frames_field"]
    interval_seconds = float(pr["frame_interval_seconds"])
    max_width = int(pr["max_image_width"])
    image_quality = int(pr["image_quality"])
    working_dir = Path(pr["working_dir"])

    print(f"\n=== Processing Trackor {trackor_id} ===")

    trackor_work_dir = working_dir / f"trackor_{trackor_id}"
    if trackor_work_dir.exists():
        shutil.rmtree(trackor_work_dir)
    trackor_work_dir.mkdir(parents=True, exist_ok=True)

    video_dir = trackor_work_dir / "video"
    frames_dir = trackor_work_dir / "frames"

    try:
        video_path = download_source_video(
            session=session,
            base_url=base_url,
            trackor_id=trackor_id,
            source_field=source_video_field,
            output_dir=video_dir
        )

        frame_paths = extract_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            trackor_id=trackor_id,
            interval_seconds=interval_seconds,
            max_width=max_width,
            jpeg_quality=image_quality
        )

        for frame_path in frame_paths:
            upload_file_to_multiefile(
                session=session,
                base_url=base_url,
                trackor_id=trackor_id,
                target_field=target_frames_field,
                file_path=frame_path
            )

        # Only reset the trigger AFTER all steps above succeeded
        reset_trigger_field(
            session=session,
            base_url=base_url,
            trackor_id=trackor_id,
            trigger_field=trigger_field,
            reset_value=0
        )

        print(f"Trackor {trackor_id} processed successfully.")

    finally:
        # Keep work directory cleanup deterministic
        if trackor_work_dir.exists():
            shutil.rmtree(trackor_work_dir, ignore_errors=True)


def main():
    try:
        settings = load_settings(SETTINGS_FILE)

        ov = settings["onevizion"]
        pr = settings["processing"]

        base_url = ov["base_url"].rstrip("/")
        access_key = ov["access_key"]
        secret_key = ov["secret_key"]

        trackor_type = pr["trackor_type"]
        trigger_field = pr["trigger_field"]

        session = create_session(access_key=access_key, secret_key=secret_key)

        trackors = get_trackors(session, base_url, trackor_type, trigger_field)
        print(f"Found {len(trackors)} triggered trackor(s).")

        if not trackors:
            print("No matching records found.")
            return

        for trackor in trackors:
            try:
                process_trackor(
                    session=session,
                    base_url=base_url,
                    trackor=trackor,
                    settings=settings
                )
            except Exception as e:
                trackor_id = get_trackor_id(trackor) or "<unknown>"
                print(f"ERROR processing Trackor {trackor_id}: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
