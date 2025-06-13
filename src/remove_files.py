import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_paths(path):
    to_delete = []
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            to_delete.append(os.path.join(root, f))
        for d in dirs:
            to_delete.append(os.path.join(root, d))
    to_delete.append(path)  # delete the root dir last
    return to_delete

def delete_path(path):
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            os.rmdir(path)
        return True
    except Exception as e:
        print(f"Could not delete {path}: {e}")
        return False

def delete_with_progress_parallel(path, max_workers=8):
    to_delete = collect_paths(path)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(delete_path, p): p for p in to_delete}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Deleting", unit="item"):
            pass  # Progress bar updates as each file is deleted

def main():
    path = "/hadatasets/alef.ferreira/VGGSound/VGGSound/image"
    delete_with_progress_parallel(path, max_workers=24)

if __name__ == "__main__":
    main()
