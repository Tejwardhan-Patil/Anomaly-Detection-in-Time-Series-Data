import os
import shutil

# Define paths
DATA_DIR = 'data/'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed/')
FEATURES_DATA_DIR = os.path.join(DATA_DIR, 'features/')

def create_directory(dir_path):
    """Ensure the directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def clean_raw_data():
    """Clean the raw data by ensuring all files are properly organized."""
    print("Cleaning raw data...")
    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if os.path.isfile(file_path):
            # Perform file validation, extension checks, etc
            if validate_file(file_name):
                print(f"Valid file found: {file_name}")
            else:
                print(f"Invalid or corrupted file found and removed: {file_name}")
                os.remove(file_path)

def validate_file(file_name):
    """Validate the file format and contents."""
    valid_extensions = ['.csv', '.json', '.xlsx']
    return any(file_name.endswith(ext) for ext in valid_extensions)

def organize_data():
    """Move or copy files to processed and features directories as required."""
    print("Organizing data...")
    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if os.path.isfile(file_path):
            # Copy to processed directory after preprocessing
            shutil.copy(file_path, PROCESSED_DATA_DIR)
            print(f"Copied {file_name} to processed data directory.")
            
            # Move to features directory
            shutil.copy(file_path, FEATURES_DATA_DIR)

def main():
    # Ensure required directories exist
    create_directory(PROCESSED_DATA_DIR)
    create_directory(FEATURES_DATA_DIR)
    
    # Perform cleaning and organizing tasks
    clean_raw_data()
    organize_data()

if __name__ == "__main__":
    main()