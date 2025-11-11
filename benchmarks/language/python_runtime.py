from time import perf_counter
import numpy as np
import magtrack

stack = np.ones((100, 100, 100), dtype=np.float64)

# Warm up
for _ in range(100):
    magtrack.center_of_mass(stack)

# Runtime measurement
start_time = perf_counter()
for _ in range(1000):
    magtrack.center_of_mass(stack)
end_time = perf_counter()
elapsed_time = end_time - start_time
print(f"Runtime: {elapsed_time:.6f} seconds")