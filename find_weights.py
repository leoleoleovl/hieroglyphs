import os

runs_dir = r"C:\Users\admin\Desktop\hieroglyph_app\runs"
for root, dirs, files in os.walk(runs_dir):
    for file in files:
        if file.endswith(".pt"):
            print(f"Found weight file at: {os.path.join(root, file)}")