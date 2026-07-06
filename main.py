from src.data_loader import load_data

data = load_data(r"C:\Users\admin\Downloads\education rs\saved_files")

print(data["training_data"].head())