import subprocess
import time
import matplotlib.pyplot as plt
import os

executable_path = "../../speed_cpu"
output_folder = "./"

start = 100000
end = 1000000
step = 100000  # Adjust the step size to control the number of points

sizes = []
times_run = []
times_run_thread = []

def run(size):
    arguments = ["../../weights_and_biases.txt", "-testmode", str(size)]
    start_time = time.time()
    subprocess.run([executable_path] + arguments, capture_output=True, text=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def run_thread(size):
    arguments = ["../../weights_and_biases.txt", "-testmode", str(size), "-t"]
    start_time = time.time()
    subprocess.run([executable_path] + arguments, capture_output=True, text=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

for size in range(start, end + 1, step):
    time_run_thread = run_thread(size)
    time_run = run(size)
    sizes.append(size/1000000)
    times_run.append(time_run)
    times_run_thread.append(time_run_thread)
    print(f"Size: {size}, Run: {time_run}, Run with Thread: {time_run_thread}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_run, label='Run', color='blue', marker='o')
plt.plot(sizes, times_run_thread, label='Run with Thread', color='red', marker='x')
plt.xlabel('Size (millions)')
plt.ylabel('Time (seconds)')
plt.title('Execution Time vs. Size')
plt.legend()
plt.grid(True)

# Save the plot to the output folder
plot_path = os.path.join(output_folder, "execution_time_vs_size.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to {plot_path}")