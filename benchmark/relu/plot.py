import os
import subprocess
import matplotlib.pyplot as plt

result = subprocess.run(['make'], capture_output=True, text=True)
# Define the folder containing the executables
folder_path = './bin'  # Change this to your bin folder path

# Define the input sizes to test
start=100000
end=1000000
step=10000


input_sizes = list(range(start, end+1, step))
# Initialize a dictionary to store runtimes for each executable
runtimes = {exe: [] for exe in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, exe))}

# Loop through each executable
for exe in runtimes.keys():
    exe_path = os.path.join(folder_path, exe)
    
    # Loop through each input size
    for n in range(start,end+1,step):
        # Run the executable with the input size and capture its output
        result = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
        
        # Parse the output to get the runtime
        runtime = float(result.stdout.strip())
        print(exe,runtime)
        
        # Append the runtime to the corresponding executable list
        runtimes[exe].append(runtime)

# Plot the data
plt.figure(figsize=(12, 6))

# Loop through each executable and plot the runtimes
for exe, times in runtimes.items():
    plt.plot(input_sizes, times, marker='o', label=exe)

plt.xlabel('Input Size')
plt.ylabel('Runtime (s)')
plt.title('Benchmark of Function Versions')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
