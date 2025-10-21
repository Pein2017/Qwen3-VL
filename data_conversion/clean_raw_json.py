import argparse
import json
import os


def clean_annotation_file(input_path, output_path, lang="both"):
    """
    Reads an annotation JSON file, extracts key information, and saves it to a new file.
    Preserves the original JSON structure while cleaning the features.

    Args:
        input_path (str): The path to the original JSON file.
        output_path (str): The path to save the cleaned JSON file.
        lang (str): The language to keep ('zh', 'en', or 'both').
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Start with the original data structure
    cleaned_data = {}

    # Preserve essential metadata sections that are needed by the pipeline
    essential_keys = ["info", "tagInfo", "version"]
    for key in essential_keys:
        if key in data:
            cleaned_data[key] = data[key]

    # Clean the features in markResult
    cleaned_features = []
    if "markResult" in data and "features" in data["markResult"]:
        for feature in data["markResult"]["features"]:
            properties: dict = {}
            original_properties: dict = feature.get("properties", {})

            # 根据语言选项保留中英文信息
            if lang in ("zh", "both"):
                properties["contentZh"] = original_properties.get("contentZh", {})
            if lang in ("en", "both"):
                properties["content"] = original_properties.get("content", {})

            # Convert legacy "Square" geometry type to "Quad" for consistency
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "Square":
                geometry = geometry.copy()
                geometry["type"] = "Quad"
                print(f"  Converted geometry type 'Square' -> 'Quad' in feature")

            cleaned_features.append(
                {
                    # GeoJSON 要求的 Feature 类型
                    "type": feature.get("type", "Feature"),
                    "geometry": geometry,
                    "properties": properties,
                }
            )

    # Preserve markResult structure with cleaned features
    if "markResult" in data:
        cleaned_data["markResult"] = {
            "features": cleaned_features,
            "type": data["markResult"].get("type", "FeatureCollection"),
        }
        # Preserve other markResult fields if they exist
        for key in data["markResult"]:
            if key not in ["features", "type"]:
                cleaned_data["markResult"][key] = data["markResult"][key]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False)


def process_folder(input_dir, output_dir, lang="both"):
    """
    Processes all JSON files in a directory, cleans them, and saves them
    to an output directory.

    Args:
        input_dir (str): The directory containing the original JSON files.
        output_dir (str): The directory where cleaned files will be saved.
        lang (str): The language to keep ('zh', 'en', or 'both').
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing {input_path} -> {output_path}")
            clean_annotation_file(input_path, output_path, lang)

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean annotation JSON files.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="The input directory containing the JSON files (e.g., 'ds').",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The output directory for the cleaned JSON files (e.g., 'ds_clean').",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="both",
        choices=["zh", "en", "both"],
        help="Language to keep: 'zh' for Chinese, 'en' for English, or 'both'. Defaults to 'both'.",
    )

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.lang)
