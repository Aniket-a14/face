import os
import pandas as pd

def parse_face_folder(root_dir):
    image_paths, genders, identities = [], [], []
    for gender_folder in ["male", "female"]:
        gender_val = 1 if gender_folder == "male" else 0
        folder_path = os.path.join(root_dir, gender_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder_path, fname))
                genders.append(gender_val)
                # Use filename (without extension) as identity
                identity = os.path.splitext(fname)[0]
                identities.append(identity)
    df = pd.DataFrame({
        "image_path": image_paths,
        "gender": genders,
        "identity": identities
    })
    return df

if __name__ == "__main__":
    input_dir = "data/raw"
    df = parse_face_folder(input_dir)
    df.to_csv("face_processed.csv", index=False)
    print("✅ Dataset processed and saved as face_processed.csv")
