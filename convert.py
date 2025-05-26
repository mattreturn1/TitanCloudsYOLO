import os
import json
from pathlib import Path

def convert_labelme_to_yolo_seg(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = list(Path(json_dir).glob("*.json"))

    for json_file in json_files:
        print(f"Processing {json_file.name}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        yolo_lines = []

        for shape in shapes:
            print(f"  shape_type: {shape.get('shape_type')}  | label: {shape.get('label')}")

            if shape["shape_type"] != "polygon":
                continue

            points = shape["points"]
            # normalize based on the image size
            if data.get("imageHeight", 1024) == 1024:
                normalized = [coord / 1024 for pt in points for coord in pt]
            else:
                normalized = [coord / 512 for pt in points for coord in pt]

            line = "0 " + " ".join(map(str, normalized))  # class 0 = "cloud"
            yolo_lines.append(line)

        out_path = Path(output_dir) / (json_file.stem + ".txt")
        with open(out_path, 'w') as f:
            f.write("\n".join(yolo_lines))

    print(f"Converted {len(json_files)} files to YOLO format at {output_dir}")

# Perform the conversion for each split
#TODO also images in new Dataset repository
if __name__ == "__main__":
    convert_labelme_to_yolo_seg("Dataset_NASA/train/labels", "Dataset/train/labels")
    convert_labelme_to_yolo_seg("Dataset_NASA/test/labels", "Dataset/test/labels")
