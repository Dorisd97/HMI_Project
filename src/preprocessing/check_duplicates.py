import os
import zipfile
import hashlib

# Step 1: Unzip the file
zip_path = 'D:/Projects/HMI/HMI_Project/data/Enron.zip'
unzip_dir = 'D:/Projects/HMI/HMI_Project/data/enron_data'

if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

# Step 2: Read all .txt files and compute their hashes
file_hashes = {}

for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
                file_hash = hashlib.md5(file_content.encode('utf-8')).hexdigest()

                if file_hash in file_hashes:
                    file_hashes[file_hash].append(file_path)
                else:
                    file_hashes[file_hash] = [file_path]

# Step 3: Find and display duplicates
duplicate_files = {hash: paths for hash, paths in file_hashes.items() if len(paths) > 1}

print(f"Total sets of duplicate files found: {len(duplicate_files)}")

for hash_value, files in duplicate_files.items():
    print("\nDuplicate set:")
    for file in files:
        print(file)

# Step 4: Remove duplicate files (keep only one file per set) and log the deleted ones
log_file_path = 'D:/Projects/HMI/HMI_Project/data/deleted_duplicates_log.txt'
with open(log_file_path, 'w') as log_file:
    for files in duplicate_files.values():
        for duplicate_file in files[1:]:  # Skip the first file, remove others
            os.remove(duplicate_file)
            log_file.write(f"Deleted duplicate file: {duplicate_file}\n")
            print(f"Deleted duplicate file: {duplicate_file}")

print(f"All deleted duplicates have been logged to {log_file_path}")
