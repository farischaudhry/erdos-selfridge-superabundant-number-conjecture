import numpy as np
import sympy as sp
import pickle
from collections import defaultdict
from numba import njit, prange

# GPU-Accelerated divisor sum function (chunked for large numbers)
@njit(parallel=True)
def fast_divisor_sigma_chunked(start, end):
    sigma = np.ones(end - start + 1, dtype=np.uint64)
    for i in prange(1, end + 1):
        for j in range(max(i, start), end + 1, i):
            sigma[j - start] += i
    return sigma

# Function to compute superabundant numbers
# A superabundant number is one where σ(n)/n is maximized compared to smaller numbers
def is_superabundant(n, prev_max, sigma, start):
    ratio = sigma[n - start] / n
    return ratio > prev_max, ratio

# Save progress to a file
def save_progress(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Load progress from a file
def load_progress(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Generate superabundant numbers up to a given limit (chunked processing)
def generate_superabundant(limit, chunk_size=10**6, progress_file="superabundant_progress.pkl"):
    progress = load_progress(progress_file)
    if progress:
        superabundant_numbers, prev_max, n = progress
        print(f"Resuming from n={n}")
    else:
        superabundant_numbers = []
        prev_max = 0
        n = 1

    while n < limit:
        chunk_end = min(n + chunk_size, limit)
        print(f"Computing divisor sums using GPU for range {n} - {chunk_end}...")
        sigma = fast_divisor_sigma_chunked(n, chunk_end)

        for num in range(n, chunk_end):
            is_sa, ratio = is_superabundant(num, prev_max, sigma, n)
            if is_sa:
                superabundant_numbers.append(num)
                prev_max = ratio
                print(f"Found superabundant number {num} with ratio {ratio}")
                save_progress(progress_file, (superabundant_numbers, prev_max, num))
            if num % 100000 == 0:
                save_progress(progress_file, (superabundant_numbers, prev_max, num))
                print(f"Saved progress at n={num}")
        n = chunk_end

    save_progress(progress_file, (superabundant_numbers, prev_max, n))
    return superabundant_numbers

# Build connectivity graph of superabundant numbers
def build_graph(superabundant_numbers):
    graph = defaultdict(set)
    sa_set = set(superabundant_numbers)
    for n in superabundant_numbers:
        for p in sp.primerange(2, 1000):  # Limit prime range for efficiency
            np_ = n * p
            nq = n // p if n % p == 0 else None
            if np_ in sa_set:
                graph[n].add(np_)
                graph[np_].add(n)
            if nq and nq in sa_set:
                graph[n].add(nq)
                graph[nq].add(n)
    return graph

# Check if the superabundant numbers form a connected graph
def check_connectivity(graph, start=1, counterexample_file="counterexamples.txt"):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)
    disconnected = set(graph.keys()) - visited
    if disconnected:
        with open(counterexample_file, "w") as f:
            f.writelines(f"{num}\n" for num in disconnected)
        print(f"Found {len(disconnected)} disconnected superabundant numbers! Possible counterexamples saved to {counterexample_file}.")
    else:
        print("Superabundant numbers are fully connected.")
    return disconnected

# Set computation limit (reduce if necessary for memory)
LIMIT = 10**20
CHUNK_SIZE = 10**6

# Generate superabundant numbers and check connectivity
superabundant_numbers = generate_superabundant(LIMIT, CHUNK_SIZE)
print(f"Generated {len(superabundant_numbers)} superabundant numbers up to {LIMIT}")

# Build graph
graph = build_graph(superabundant_numbers)

# Check connectivity
check_connectivity(graph)
