import kagglehub

print("Starting download from Kaggle... Please wait...")
path = kagglehub.dataset_download("waleedumer/egyptian-hieroglyphics-datasets")
print("--------------------------------------------------")
print("SUCCESS! Your dataset downloaded to this location:")
print(path)
print("--------------------------------------------------")