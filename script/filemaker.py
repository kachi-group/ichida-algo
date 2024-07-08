import os
import shutil
import argparse

def generate_files(src_folder, start, end, target_total):
    current_max = end
    while current_max < target_total:
        for i in range(start, end + 1):
            src_file = os.path.join(src_folder, f"{i:02d}out.txt")
            if not os.path.exists(src_file):
                print(f"Source file {src_file} does not exist. Exiting.")
                return
            current_max += 1
            new_file = os.path.join(src_folder, f"{current_max:02d}out.txt")
            shutil.copy(src_file, new_file)
            if current_max >= target_total:
                break
    print(f"Generated up to {current_max} files.")

def cleanup_files(src_folder, start, end, target_total):
    for i in range(end + 1, target_total + 1):
        file_to_remove = os.path.join(src_folder, f"{i:02d}out.txt")
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
    print(f"Cleaned up files from {end + 1} to {target_total}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally clean up files.")
    parser.add_argument("--src-folder", type=str, required=True, help="Source folder containing original files.")
    parser.add_argument("--total", type=int, required=True, help="Total number of files to generate.")
    parser.add_argument("--cleanup", action="store_true", help="Clean up generated files.")
    
    args = parser.parse_args()
    
    start = 1
    end = 52

    if args.cleanup:
        cleanup_files(args.src_folder, start, end, args.total)
    else:
        generate_files(args.src_folder, start, end, args.total)