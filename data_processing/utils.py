import os
import json
import hashlib
import pandas as pd

from PIL import Image
from datasets import load_dataset

def load_image(image_file):
    return Image.open(image_file).convert("RGB")

def make_unique_id(long_id, question_text):
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"

def load_all_vqa_pairs(vqa_file, dataset_name='wsi_vqa', image_dir=None):
    """
    General VQA data loading function.
    Supports:
      - WSI-VQA
      - SlideBench-VQA (TCGA)
      - local_wsi
    Returns:
      vqa_pairs: list[dict]
        {
            "short_id": Optional short ID,
            "long_id": Filename or full path,
            "question": Question,
            "choices": Choices (if available, else None),
            "answer": Answer,
            "image": Image object (if available)
        }
    """
    assert os.path.exists(vqa_file), f"File not found: {vqa_file}"
    vqa_pairs = []

    # ---------- 1. WSI-VQA ----------
    if dataset_name.lower() == 'wsi_vqa':
        with open(vqa_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        case2qas = {}
        for item in data:
            sid = item["Id"]
            case2qas.setdefault(sid, []).append(item)

        if image_dir is not None:
            dirs = [d for d in os.listdir(image_dir) if "DX1" in d]
        else:
            dirs = case2qas.keys()

        for d in dirs:
            long_id = d if image_dir else None
            short_id = d[:12] if image_dir else d

            if short_id not in case2qas:
                continue

            for qa in case2qas[short_id]:
                vqa_pairs.append({
                    "short_id": short_id,
                    "long_id": long_id,
                    "question": qa["Question"],
                    "choices": qa.get("Choice"),
                    "answer": qa["Answer"]
                })

    # ---------- 2. SlideBench-VQA (TCGA) ----------
    elif dataset_name.lower() == 'slidebench_vqa':
        df = pd.read_csv(vqa_file)

        if image_dir is not None:
            valid_slides = set([d.split(".")[0] for d in os.listdir(image_dir) if "DX1" in d])
            df = df[df["Slide"].isin(valid_slides)]

        for _, row in df.iterrows():
            choices = [row["A"], row["B"], row["C"], row["D"]]
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            ans_letter = str(row["Answer"]).strip().upper()
            ans_text = choices[answer_map[ans_letter]] if ans_letter in answer_map else row["Answer"]

            vqa_pairs.append({
                "long_id": row["Slide"],
                "question": row["Question"],
                "choices": choices,
                "answer": ans_text
            })

    # ---------- 3. Local WSI ----------
    elif dataset_name.lower() == 'local_wsi':
        if vqa_file.lower().endswith(".csv"):
            data = pd.read_csv(vqa_file).to_dict(orient="records")
        else:
            with open(vqa_file, "r", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, dict):
            data = data.get("questions", [])

        available_slides = None
        if image_dir is not None and os.path.isdir(image_dir):
            available_slides = {d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))}

        for item in data:
            long_id = item.get("slide_id") or item.get("long_id") or item.get("Slide") or item.get("Id")
            if not long_id:
                raise ValueError(f"local_wsi item missing slide_id/long_id: {item}")
            if available_slides is not None and long_id not in available_slides:
                continue

            choices = item.get("choices", item.get("Choice"))
            if isinstance(choices, str):
                try:
                    parsed_choices = json.loads(choices)
                    choices = parsed_choices
                except json.JSONDecodeError:
                    choices = [x.strip() for x in choices.split("|") if x.strip()]

            vqa_pairs.append({
                "short_id": item.get("short_id", long_id),
                "long_id": long_id,
                "case_id": item.get("case_id") or item.get("short_id") or long_id.split("_")[0],
                "question_id": item.get("question_id"),
                "question": item.get("question") or item.get("Question"),
                "question_type": item.get("question_type"),
                "difficulty": item.get("difficulty"),
                "evidence_tier": item.get("evidence_tier", "strict"),
                "option_concepts": item.get("option_concepts"),
                "choices": choices,
                "answer": item.get("answer", item.get("Answer", "")),
                "answer_zh": item.get("answer_zh"),
                "source_report_fields": item.get("source_report_fields"),
                "source_report_evidence": item.get("source_report_evidence"),
            })

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    print(f"Loaded {len(vqa_pairs)} VQA pairs from {dataset_name}")
    return vqa_pairs


def extract_coords_from_name(patch_name: str):
    """
    Extract coordinates (x, y) from patch filename, e.g., '37856_28960.jpg' -> (37856, 28960)
    """
    base = os.path.basename(patch_name)
    name_no_ext = os.path.splitext(base)[0]
    try:
        x_str, y_str, *_ = name_no_ext.split("_")
        return int(x_str), int(y_str)
    except Exception:
        return None, None

def build_descriptions_with_meta(items, mag_level=None, include_header=True, include_coords=True):
    """
    Combine multiple (patch_name, description) into text for LLM:
      - If include_header=True, inserts "[Current Magnification: {mag_level}x]" at the top
      - Each patch is annotated with coordinates (extracted from filename); if include_coords=False, coordinates are omitted
      - Output format example:
        [Current Magnification: 10x]
        [37856_28960 | Coord=(37856,28960)] description...
        [82912_86304 | Coord=(82912,86304)] description...
    """
    header = ""
    if include_header and mag_level is not None:
        header = f"[Current Magnification: {mag_level}x]\n\n"

    parts = []
    for name, desc in items:
        x, y = extract_coords_from_name(name)
        coord_str = f"({x},{y})" if (x is not None and y is not None) else "(unknown)"
        if include_coords:
            parts.append(f"[{name} | Coord={coord_str}] {desc}")
        else:
            parts.append(f"[{name}] {desc}")

    body = "\n\n".join(parts)
    return header + body

def get_patch_fullpath(PATCH_ROOT, TARGET_LONG_ID, patch_name):
    """
    Try to return the full path of the patch (if not exists, try adding common extensions)
    """
    base = os.path.join(PATCH_ROOT, TARGET_LONG_ID, patch_name)
    if os.path.exists(base):
        return base
    for ext in [".jpg", ".jpeg", ".png"]:
        p = base + ext
        if os.path.exists(p):
            return p
    return base

def split_patch_for_zoom(patch_path, zoom_level, source_magnification=5):
    """
    Split a 5x patch image into sub-images at a higher magnification level,
    and return the sub-images along with their global coordinates.
    Supports patch_path as a file path or a PIL.Image.Image object.
    """

    if isinstance(patch_path, (str, os.PathLike)):
        img = Image.open(patch_path).convert("RGB")
        base_name = os.path.basename(patch_path)
        name_no_ext = os.path.splitext(base_name)[0]

    elif isinstance(patch_path, Image.Image):
        img = patch_path
        # If it is an image object, use default coordinates (0,0)
        name_no_ext = "0_0"
    else:
        raise TypeError(f"patch_path must be a file path or PIL.Image.Image object, received {type(patch_path)}")

    width, height = img.size

    # --- Check if zoom_level is valid ---
    if source_magnification < 5 or zoom_level <= source_magnification:
        raise ValueError(
            "zoom_level must be greater than source_magnification "
            f"(received source={source_magnification}, target={zoom_level})"
        )
    if zoom_level % source_magnification != 0:
        raise ValueError(
            f"zoom_level must be an integer multiple of source_magnification, received {zoom_level}/{source_magnification}"
        )

    # --- Try to parse global coordinates from filename ---
    try:
        parts = name_no_ext.split("_")
        base_x, base_y = int(parts[0]), int(parts[1])
    except ValueError:
        base_x, base_y = 0, 0  # If parsing fails, default to starting from (0, 0)

    # --- Calculate split factor ---
    factor = zoom_level // source_magnification

    # --- Calculate sub-image size ---
    sub_w = width // factor
    sub_h = height // factor

    # --- Split image and calculate global coordinates ---
    patches = []
    for i in range(factor):
        for j in range(factor):
            left = j * sub_w
            upper = i * sub_h
            right = (j + 1) * sub_w if j < factor - 1 else width
            lower = (i + 1) * sub_h if i < factor - 1 else height
            patch = img.crop((left, upper, right, lower))

            # Calculate global coordinates of the sub-image
            global_x = base_x + left
            global_y = base_y + upper

            patches.append((patch, (global_x, global_y)))

    return patches

def get_specific_case_descriptions(file_path, long_id):
    print(f"Searching for long ID: {long_id} in {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        descriptions_dict = data.get(long_id)
        if descriptions_dict and isinstance(descriptions_dict, dict):
            print(f"Successfully found and extracted dictionary containing {len(descriptions_dict)} descriptions.")
            return descriptions_dict
        else:
            print(f"Error: ID '{long_id}' not found in file or format is not a dictionary.")
            return None
    except Exception as e:
        print(f"Error: Error reading description file - {e}")
        return None
