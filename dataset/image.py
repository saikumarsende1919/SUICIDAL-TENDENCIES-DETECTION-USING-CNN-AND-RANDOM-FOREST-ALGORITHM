import kagglehub

# Download latest version
path = kagglehub.dataset_download("emotion/dataset/fer2013")

print("Path to dataset files:", path)