import os
import glob
import pandas as pd

def parse_utkface(directory):
    image_paths, genders, ids = [], [], []
    for path in glob.glob(os.path.join(directory, "*.jpg.chip.jpg")):
        filename = os.path.basename(path)
        parts = filename.split("_")
        try:
            gender = int(parts[1])
            identity = int(parts[3].split('.')[0])
            image_paths.append(path)
            genders.append(gender)
            ids.append(identity)
        except (IndexError, ValueError):
            continue
    df = pd.DataFrame({
        'image_path': image_paths,
        'gender': genders,
        'identity': ids
    })
    return df

if __name__ == "__main__":
    input_dir = "data/raw/UTKFace"
    df = parse_utkface(input_dir)
    df.to_csv("utkface_processed.csv", index=False)
    print("✅ Dataset processed and saved as utkface_processed.csv")
