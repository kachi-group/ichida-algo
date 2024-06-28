import os
import struct

def read_and_convert_file(file_path, output_file):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        float_values = [float(x) for x in content.split(',')]
        
        with open(output_file, 'wb') as bin_file:
            for value in float_values:
                bin_file.write(struct.pack('f', value))

if __name__ == "__main__":
    directory = "./txttensors"  # Replace with the path to your directory
    
    for i in range(1, 53):  # Assuming files are named 01out.txt to 52out.txt
        file_name = f"{i:02d}out.txt"
        file_path = os.path.join(directory, file_name)
        output_file = os.path.join(directory, f"{i:02d}out.bin")
        
        read_and_convert_file(file_path, output_file)
        print(f"Binary file saved to {output_file}")
