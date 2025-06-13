import argparse
import subprocess
import os
import glob
import time
import shutil
import random
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path

# --- Helper Functions ---

def is_tool_available(name):
    """Check whether a command-line tool is available in the system's PATH."""
    return shutil.which(name) is not None

def compile_cpp(source_file, executable_path, compiler_binary, flags):
    """Compiles a C++ source file into an executable."""
    compile_command = [compiler_binary, "-O3", "-std=c++17", "-w"] + flags + [source_file, "-o", executable_path]
    print(f"Compiling {source_file} -> {executable_path}...")
    print(f"Command: {' '.join(compile_command)}")
    try:
        process = subprocess.run(compile_command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            print(f"Error compiling {source_file}:\n--- STDERR ---\n{process.stderr}\n--- STDOUT ---\n{process.stdout}")
            return False
        print(f"Successfully compiled {executable_path}")
        return True
    except Exception as e:
        print(f"An exception occurred during compilation: {e}")
        return False

def run_task(task_details, result_queue):
    """
    Runs a single benchmark task, pins it to a CPU core using taskset,
    and saves its output to files.
    """
    executable_path = task_details["executable_path"]
    graph_file = task_details["graph_file"]
    core_id = task_details["core_id"]
    time_limit_sec = task_details["time_limit_sec"]
    seed = task_details["seed"]
    output_dir = task_details["output_dir"]

    # Timeout for the script's subprocess call should be slightly longer than the program's internal limit
    script_timeout = time_limit_sec * 1.5

    # --- Construct the command to be executed ---
    # Command format: taskset -c [core] [executable] [graph] [time_limit] [seed]
    command = [
        "taskset", "-c", str(core_id),
        executable_path,
        graph_file,
        str(time_limit_sec),
        str(seed)
    ]

    graph_basename = Path(graph_file).stem
    stdout_path = output_dir / f"{graph_basename}_seed{seed}.out"
    stderr_path = output_dir / f"{graph_basename}_seed{seed}.err"
    
    result = {
        "graph_file": graph_file,
        "status": "started",
        "command": " ".join(command)
    }

    print(f"Running on core {core_id}: {Path(executable_path).name} on {Path(graph_file).name} with seed {seed}")

    try:
        start_time = time.time()
        process = subprocess.run(command, capture_output=True, text=True, timeout=script_timeout, check=False)
        end_time = time.time()

        # Save stdout and stderr to their respective files
        with open(stdout_path, 'w') as f:
            f.write(process.stdout)
        if process.stderr:
            with open(stderr_path, 'w') as f:
                f.write(process.stderr)

        result["script_wall_time_sec"] = round(end_time - start_time, 3)
        
        if process.returncode == 0:
            result["status"] = "completed"
        else:
            result["status"] = f"crashed (code: {process.returncode})"

    except subprocess.TimeoutExpired:
        result["status"] = "timed_out"
        # Create an error file indicating timeout
        with open(stderr_path, 'w') as f:
            f.write(f"Process timed out after {script_timeout} seconds.")
            
    except Exception as e:
        result["status"] = f"script_error ({e})"
        with open(stderr_path, 'w') as f:
            f.write(f"The script encountered an error trying to run the subprocess: {e}")
            
    finally:
        result_queue.put(result)
        print(f"Finished on core {core_id}: {Path(graph_file).name}. Status: {result['status']}")


# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(
        description="A simple C++ benchmark runner using taskset for CPU pinning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_file", help="Path to the C++ source file to be compiled and tested.")
    parser.add_argument("graph_folder", help="Path pattern to graph files (e.g., 'data/*.clq').")
    parser.add_argument("output_dir", help="Directory to save the stdout and stderr log files.")
    parser.add_argument("time_limit", type=int, help="Time limit in seconds for each program run.")
    parser.add_argument("--target_bin_name", default="solver.out", help="Name for the compiled binary file.")
    parser.add_argument("--max_concurrent_runs", type=int, default=0, help="Max concurrent runs. Set to 0 to use all available CPU cores.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for the solver. If not set, a random seed is used for each run.")
    parser.add_argument("--compiler_choice", choices=['gcc', 'llvm'], default='gcc', help="Compiler to use (g++ or clang++).")
    parser.add_argument("--compiler_flags", nargs='*', default=[], help="Additional flags to pass to the C++ compiler.")
    args = parser.parse_args()

    # --- Pre-run checks and setup ---
    if not is_tool_available("taskset"):
        print("Error: 'taskset' command not found. CPU pinning is not possible. Exiting.")
        return

    compiler_path = "g++" if args.compiler_choice == 'gcc' else "clang++"
    if not is_tool_available(compiler_path):
        print(f"Error: Compiler '{compiler_path}' not found. Exiting.")
        return
        
    build_dir = Path("build"); build_dir.mkdir(exist_ok=True)
    executable_path = build_dir / args.target_bin_name

    output_dir_path = Path(args.output_dir); output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir_path.resolve()}")

    # --- Compile the target program ---
    if not compile_cpp(args.source_file, str(executable_path), compiler_path, args.compiler_flags):
        print("Compilation failed. Please check the source file and compiler flags.")
        return
    
    executable_abs_path = str(executable_path.resolve())

    # --- Discover graph files ---
    graph_files = glob.glob(args.graph_folder)
    if not graph_files:
        print(f"Error: No files found matching the pattern: {args.graph_folder}")
        return
    print(f"Found {len(graph_files)} graph files to process.")

    # --- Generate tasks ---
    tasks_to_run = []
    for graph_path in graph_files:
        task_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
        task = {
            "executable_path": executable_abs_path,
            "graph_file": str(Path(graph_path).resolve()),
            "time_limit_sec": args.time_limit,
            "seed": task_seed,
            "output_dir": output_dir_path.resolve()
        }
        tasks_to_run.append(task)
    
    if not tasks_to_run:
        print("No tasks were generated. Exiting.")
        return

    # --- Execute tasks concurrently ---
    result_q = Queue()
    num_cores = cpu_count()
    max_procs = args.max_concurrent_runs if args.max_concurrent_runs > 0 else num_cores
    print(f"Using up to {max_procs} concurrent processes on {num_cores} available cores.")

    active_procs = []
    available_cores = list(range(num_cores))
    tasks_to_do = list(tasks_to_run)
    total_tasks = len(tasks_to_do)

    while tasks_to_do or active_procs:
        # 1. Clean up finished processes and free their cores
        for i in range(len(active_procs) - 1, -1, -1):
            proc, core_id = active_procs[i]
            if not proc.is_alive():
                proc.join()
                active_procs.pop(i)
                available_cores.append(core_id)
        
        # 2. Check for results from the queue
        while not result_q.empty():
            # In this version, we just consume the item to track progress.
            result_q.get()
            completed_count = total_tasks - len(tasks_to_do) - len(active_procs)
            print(f"Progress: {completed_count}/{total_tasks} tasks completed.")

        # 3. Launch new processes if there are tasks and available cores
        while tasks_to_do and len(active_procs) < max_procs and available_cores:
            task_spec = tasks_to_do.pop(0)
            core_to_assign = available_cores.pop(0)
            
            task_spec["core_id"] = core_to_assign
            p = Process(target=run_task, args=(task_spec, result_q))
            p.start()
            active_procs.append((p, core_to_assign))
        
        time.sleep(0.5) # Prevent busy-waiting

    print("\nAll benchmark tasks have concluded.")

if __name__ == "__main__":
    main()