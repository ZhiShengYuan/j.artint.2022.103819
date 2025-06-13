import argparse
import subprocess
import os
import glob
import json
import time
import re
import psutil
from multiprocessing import Process, Queue, cpu_count as mp_cpu_count
from pathlib import Path
import random
import shutil

# --- Configuration ---
AUTHOR_SOLVER_NAME = "author_solver"
MWDS_DEEPOPT_SOLVER_NAME = "mwds_deepopt_solver"

COMPILER_BINARIES = {
    "gcc": "g++",
    "llvm": "clang++"
}
BASE_COMPILE_FLAGS = {
    AUTHOR_SOLVER_NAME: ["-O3", "-std=c++11", "-w"],
    MWDS_DEEPOPT_SOLVER_NAME: ["-O3", "-std=c++17", "-pthread", "-w"],
}

DEFAULT_SEED = 12345
LOW_MEMORY_THRESHOLD_PERCENT = 50.0
HIGH_MEMORY_THRESHOLD_PERCENT = 80.0
SHORT_LAUNCH_GAP_SECONDS = 5
LONG_LAUNCH_GAP_SECONDS = 120
MAX_MEMORY_CEILING_PERCENT = 90.0
POST_GAP_HIGH_MEM_WAIT_SECONDS = 1.0

PROGRAM_INTERNAL_TIME_LIMIT_REDUCTION_SEC = 5
SCRIPT_SUBPROCESS_TIMEOUT_FACTOR = 1.5
VALGRIND_ERROR_REPORT_CODE = 77

# --- Helper Functions ---

def is_tool_available(name):
    return shutil.which(name) is not None

def get_cpu_counts():
    try:
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            print("Warning: psutil.cpu_count(logical=False) returned None. Assuming physical_cores = logical_cores.")
            physical_cores = logical_cores
        return logical_cores, physical_cores
    except TypeError:
        print("Warning: Your psutil version is outdated. SMT-awareness may be limited. Please upgrade psutil.")
        logical_cores = mp_cpu_count()
        return logical_cores, logical_cores
    except Exception as e:
        print(f"Error getting CPU counts: {e}. Falling back.")
        logical_cores = mp_cpu_count()
        return logical_cores, logical_cores

def compile_cpp(source_file, executable_name, compiler_binary, flags):
    compile_command = [compiler_binary, source_file, "-o", executable_name] + flags
    print(f"Compiling {source_file} -> {executable_name} using {compiler_binary}...")
    print(f"Command: {' '.join(compile_command)}")
    try:
        process = subprocess.run(compile_command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            print(f"Error compiling {source_file}:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")
            return False, process.stdout + "\n" + process.stderr
        print(f"Successfully compiled {executable_name}")
        return True, process.stdout + "\n" + process.stderr
    except Exception as e:
        print(f"Exception during compilation of {source_file}: {e}")
        return False, str(e)

def parse_solver_json_output(stdout_str):
    """
    Parses the JSON output from either solver.
    Expects a JSON structure like:
    {
      "results": {
        "cost": ...,
        "solution_size": ...,
        "solve_time_s": ...,
        "feasible_on_original": ...
      },
      ...
    }
    """
    data = json.loads(stdout_str)
    results_data = data.get("results", {})

    cost = results_data.get("cost")
    size = results_data.get("solution_size")
    prog_time = results_data.get("solve_time_s")
    is_feasible = results_data.get("feasible_on_original")

    parsed_data = {}
    if cost is not None:
        try:
            parsed_data["solution_cost"] = int(cost)
        except (ValueError, TypeError):
            # Let it be None if conversion fails, or raise error
            pass # Or handle more gracefully
    if size is not None:
        try:
            parsed_data["solution_size"] = int(size)
        except (ValueError, TypeError):
            pass
    if prog_time is not None:
        try:
            parsed_data["program_time_reported_sec"] = float(prog_time)
        except (ValueError, TypeError):
            pass
    if is_feasible is not None:
        parsed_data["is_feasible"] = bool(is_feasible)

    # Check for essential fields to consider the parse successful for comparison logic
    if "solution_cost" in parsed_data and "solution_size" in parsed_data:
        return parsed_data
    return None


def parse_valgrind_log(log_file_path):
    metrics = {
        "definitely_lost_bytes": 0, "definitely_lost_blocks": 0,
        "indirectly_lost_bytes": 0, "indirectly_lost_blocks": 0,
        "possibly_lost_bytes": 0, "possibly_lost_blocks": 0,
        "still_reachable_bytes": 0, "still_reachable_blocks": 0,
        "error_summary_count": 0,
        "valgrind_errors_found": False,
        "raw_summary_lines": [],
        "parse_error_message": None
    }
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped: continue
                is_summary_line = False
                if "== LEAK SUMMARY:" in line_stripped: is_summary_line = True
                elif "definitely lost:" in line_stripped:
                    is_summary_line = True
                    parts = re.findall(r"([\d,]+)\s+bytes in ([\d,]+)\s+blocks", line_stripped)
                    if parts:
                        metrics["definitely_lost_bytes"] = int(parts[0][0].replace(",", ""))
                        metrics["definitely_lost_blocks"] = int(parts[0][1].replace(",", ""))
                elif "indirectly lost:" in line_stripped:
                    is_summary_line = True
                    parts = re.findall(r"([\d,]+)\s+bytes in ([\d,]+)\s+blocks", line_stripped)
                    if parts:
                        metrics["indirectly_lost_bytes"] = int(parts[0][0].replace(",", ""))
                        metrics["indirectly_lost_blocks"] = int(parts[0][1].replace(",", ""))
                elif "possibly lost:" in line_stripped:
                    is_summary_line = True
                    parts = re.findall(r"([\d,]+)\s+bytes in ([\d,]+)\s+blocks", line_stripped)
                    if parts:
                        metrics["possibly_lost_bytes"] = int(parts[0][0].replace(",", ""))
                        metrics["possibly_lost_blocks"] = int(parts[0][1].replace(",", ""))
                elif "still reachable:" in line_stripped:
                    is_summary_line = True
                    parts = re.findall(r"([\d,]+)\s+bytes in ([\d,]+)\s+blocks", line_stripped)
                    if parts:
                        metrics["still_reachable_bytes"] = int(parts[0][0].replace(",", ""))
                        metrics["still_reachable_blocks"] = int(parts[0][1].replace(",", ""))
                elif "== ERROR SUMMARY:" in line_stripped:
                    is_summary_line = True
                    parts = re.findall(r"(\d+)\s+errors from", line_stripped)
                    if parts:
                        metrics["error_summary_count"] = int(parts[0])
                if is_summary_line: metrics["raw_summary_lines"].append(line_stripped)
        if (metrics["definitely_lost_bytes"] > 0 or metrics["indirectly_lost_bytes"] > 0 or \
            metrics["possibly_lost_bytes"] > 0 or metrics["error_summary_count"] > 0):
            metrics["valgrind_errors_found"] = True
    except FileNotFoundError:
        metrics["parse_error_message"] = "Valgrind log file not found."
        metrics["valgrind_errors_found"] = True # Treat as error if log is missing
    except Exception as e:
        metrics["parse_error_message"] = f"Error parsing Valgrind log: {str(e)}"
        metrics["valgrind_errors_found"] = True # Treat as error
    return metrics

def run_benchmark_task(task_details, result_queue):
    program_name = task_details["program_name"]
    executable_path = task_details["executable_path"]
    graph_file = task_details["graph_file"]
    core_id = task_details["core_id"]
    current_round_num = task_details.get("round_number", 0)
    solver_base_time_limit = task_details["time_limit_seconds"]
    
    actual_seed_used = task_details["seed_to_use"] # Using the new general seed key

    program_internal_time_limit_arg = solver_base_time_limit - PROGRAM_INTERNAL_TIME_LIMIT_REDUCTION_SEC
    if program_internal_time_limit_arg <= 0: program_internal_time_limit_arg = 1

    script_actual_subprocess_timeout = solver_base_time_limit * SCRIPT_SUBPROCESS_TIMEOUT_FACTOR
    if script_actual_subprocess_timeout <= program_internal_time_limit_arg + 5:
        script_actual_subprocess_timeout = program_internal_time_limit_arg + 10
    if script_actual_subprocess_timeout < 15: script_actual_subprocess_timeout = 15

    base_solver_cmd_args = []
    if program_name == AUTHOR_SOLVER_NAME:
        # Args: graph_file, time_limit, seed, --json
        base_solver_cmd_args = [graph_file, str(program_internal_time_limit_arg), str(actual_seed_used), "--json"]
    elif program_name == MWDS_DEEPOPT_SOLVER_NAME:
        # Args: graph_file, time_limit, num_threads, seed, --stdout
        num_threads_for_mwds = "1" # Hardcoded for now, as per previous implicit behavior for this arg slot
        base_solver_cmd_args = [graph_file, str(program_internal_time_limit_arg), num_threads_for_mwds, str(actual_seed_used), "--stdout"]
    else:
        result_queue.put({"status": "error_unknown_program", "stderr": f"Unknown program: {program_name}", "graph_file": graph_file, "program_name": program_name})
        return

    current_cmd_list = [executable_path] + base_solver_cmd_args
    
    valgrind_log_file_path_str = None
    using_valgrind = task_details.get("use_valgrind", False)
    using_systemd_run = task_details.get("use_systemd_run", False)

    if using_valgrind:
        valgrind_log_dir = Path(task_details.get("valgrind_log_dir", "valgrind_logs"))
        sanitized_graph_name = re.sub(r'[^\w_.-]', '_', Path(graph_file).name)
        # Include program name in log file for clarity, especially if seeds could be the same for different programs
        log_file_name = f"{sanitized_graph_name}_{program_name}_r{current_round_num}_s{actual_seed_used}.vg.log"
        valgrind_log_file_path_str = str(valgrind_log_dir / log_file_name)
        valgrind_cmd_prefix = [
            "valgrind", "--leak-check=full", "--show-leak-kinds=all",
            f"--error-exitcode={VALGRIND_ERROR_REPORT_CODE}", f"--log-file={valgrind_log_file_path_str}"
        ]
        current_cmd_list = valgrind_cmd_prefix + current_cmd_list

    if using_systemd_run:
        solver_memory_limit_mb = task_details.get("solver_memory_limit_mb", 1024)
        systemd_run_cmd_prefix = [
            "systemd-run", "--user", "--wait", "--collect",
            "-p", f"MemoryMax={solver_memory_limit_mb}M",
            "-p", f"CPUAffinity={core_id}"
        ]
        current_cmd_list = systemd_run_cmd_prefix + current_cmd_list
    else:
        if is_tool_available("taskset"):
             current_cmd_list = ["taskset", "-c", str(core_id)] + current_cmd_list
        
    result = {
        "graph_file": graph_file, "program_name": program_name, "core_id": core_id,
        "round_number": current_round_num, "seed_used": actual_seed_used, "status": "started",
        "solution_size": None, "solution_cost": None, "program_time_reported_sec": None,
        "is_feasible": None, # Added new field
        "script_wall_time_sec": None, "stdout": "", "stderr": "",
        "valgrind_log_file": valgrind_log_file_path_str if using_valgrind else None,
        "valgrind_metrics": None if not using_valgrind else {},
        "command_executed": " ".join(current_cmd_list)
    }
    log_round_display = current_round_num + 1
    
    wrapper_info = []
    if using_systemd_run: wrapper_info.append(f"systemd-run(MemMax:{task_details.get('solver_memory_limit_mb')}M,Core:{core_id})")
    elif is_tool_available("taskset"): wrapper_info.append(f"taskset(Core:{core_id})")
    if using_valgrind: wrapper_info.append("valgrind")
    wrapper_str = f" wrapped by [{', '.join(wrapper_info)}]" if wrapper_info else ""

    print(f"R{log_round_display}| Starting: {program_name} on {Path(graph_file).name} ({Path(executable_path).name}{wrapper_str}, "
          f"solver_arg_TL {program_internal_time_limit_arg}s, script_subprocess_TL {script_actual_subprocess_timeout:.1f}s, "
          f"seed: {actual_seed_used})")

    start_time = time.time()
    process_obj = None
    try:
        process_obj = subprocess.run(current_cmd_list, capture_output=True, text=True, timeout=script_actual_subprocess_timeout, check=False, cwd=None)
        result["stdout"] = process_obj.stdout
        result["stderr"] = process_obj.stderr
        
        valgrind_reported_error_via_exit_code = False
        if using_valgrind and process_obj.returncode == VALGRIND_ERROR_REPORT_CODE:
            valgrind_reported_error_via_exit_code = True
            result["status"] = "valgrind_error_exitcode"

        if process_obj.returncode == 0:
            parsed_output = None
            try:
                # Both solvers now output JSON and can be parsed by the same function
                parsed_output = parse_solver_json_output(process_obj.stdout)
            except json.JSONDecodeError as je:
                result["status"] = "json_parse_error"
                result["stderr"] = (result.get("stderr","") + f"\nFailed to parse JSON output: {je}\nSTDOUT was:\n{process_obj.stdout}").strip()
                print(f"R{log_round_display}| JSON Parse Error for {program_name} on {Path(graph_file).name}. Check solver's JSON output format.")
            except Exception as e: # Catch other potential errors during parsing (e.g. KeyError if structure is unexpected)
                result["status"] = "output_parse_logic_error"
                result["stderr"] = (result.get("stderr","") + f"\nError in parsing logic after JSON decode: {e}\nSTDOUT was:\n{process_obj.stdout}").strip()
                print(f"R{log_round_display}| Output Parse Logic Error for {program_name} on {Path(graph_file).name}: {e}")

            if parsed_output:
                result.update(parsed_output) # This will add solution_size, solution_cost, program_time_reported_sec, is_feasible
                result["status"] = "completed"
            # If parsed_output is None due to missing essential fields but no exception, or if an exception occurred above:
            elif result["status"] not in ["json_parse_error", "output_parse_logic_error", "valgrind_error_exitcode"]:
                # This case means parse_solver_json_output returned None (e.g. cost/size missing)
                # or some other path led to parsed_output being None without setting specific error status.
                result["status"] = "parse_error_missing_fields"
                print(f"R{log_round_display}| Output Parse Error (missing fields) for {program_name} on {Path(graph_file).name}.")
        
        elif not valgrind_reported_error_via_exit_code: 
             result["status"] = "crashed"
             print(f"R{log_round_display}| CRASH/ERROR from subprocess for {program_name} on {Path(graph_file).name}. ExitCode: {process_obj.returncode}")
             print(f"R{log_round_display}| ---- STDERR START ----")
             print(process_obj.stderr if process_obj.stderr else "<No stderr captured>")
             print(f"R{log_round_display}| ---- STDERR END ----")
             if process_obj.stdout:
                 print(f"R{log_round_display}| ---- STDOUT START ----")
                 print(process_obj.stdout)
                 print(f"R{log_round_display}| ---- STDOUT END ----")
        
        print(f"R{log_round_display}| Info: {program_name} on {Path(graph_file).name} finished. ExitCode={process_obj.returncode}. InitialStatus={result['status']}.")

    except subprocess.TimeoutExpired:
        result["status"] = "timed_out_script"
        result["stderr"] = (result.get("stderr","") + f"\nScript-level subprocess timeout after {script_actual_subprocess_timeout:.1f}s").strip()
        print(f"R{log_round_display}| Warning: {program_name} on {Path(graph_file).name} timed out (script_TL: {script_actual_subprocess_timeout:.1f}s)")
    except Exception as e:
        result["status"] = "error_script_exception"
        result["stderr"] = (result.get("stderr","") + f"\nException running command: {e}").strip()
        print(f"R{log_round_display}| Exception running {program_name} on {Path(graph_file).name}: {e}")
    finally:
        end_time = time.time()
        result["script_wall_time_sec"] = round(end_time - start_time, 3)

        if using_valgrind and valgrind_log_file_path_str:
            valgrind_metrics = parse_valgrind_log(valgrind_log_file_path_str)
            result["valgrind_metrics"] = valgrind_metrics
            if valgrind_metrics.get("valgrind_errors_found", False):
                # Preserve more specific error statuses if they already indicate a problem
                if result["status"] not in ["timed_out_script", "error_script_exception", "valgrind_error_exitcode", "crashed", "json_parse_error", "output_parse_logic_error"]:
                    result["status"] = "valgrind_errors_found_in_log"
                print(f"R{log_round_display}| Valgrind found errors/leaks for {program_name} on {Path(graph_file).name}. Log: {valgrind_log_file_path_str}")
        
        result_queue.put(result)
        print(f"R{log_round_display}| Finished: {program_name} on {Path(graph_file).name}, FinalStatus: {result['status']}, ScriptWallTime: {result['script_wall_time_sec']}s")


def generate_comparison_summary(current_round_runs):
    summary = {}
    runs_by_graph = {}
    for run in current_round_runs:
        graph = run["graph_file"]
        if graph not in runs_by_graph: runs_by_graph[graph] = {}
        
        is_clean_run = True
        if run.get("use_valgrind", False):
            valgrind_metrics = run.get("valgrind_metrics")
            if valgrind_metrics and valgrind_metrics.get("valgrind_errors_found", False):
                is_clean_run = False
        
        # Exclude runs with valgrind errors from summary (if valgrind was used)
        if not is_clean_run and run.get("use_valgrind", False):
            continue

        if run["status"] == "completed" and run["solution_cost"] is not None:
            time_to_use = run["program_time_reported_sec"] if run["program_time_reported_sec"] is not None else run["script_wall_time_sec"]
            runs_by_graph[graph][run["program_name"]] = {
                "cost": run["solution_cost"], "time": time_to_use, "size": run.get("solution_size"),
                "is_feasible": run.get("is_feasible") # Carry feasibility along
            }

    for graph, programs_data in runs_by_graph.items():
        summary[graph] = {}
        author_data = programs_data.get(AUTHOR_SOLVER_NAME)
        mwds_data = programs_data.get(MWDS_DEEPOPT_SOLVER_NAME)
        summary[graph][AUTHOR_SOLVER_NAME] = author_data
        summary[graph][MWDS_DEEPOPT_SOLVER_NAME] = mwds_data
        winner_cost, winner_time = "N/A", "N/A"
        if author_data and mwds_data:
            if author_data["cost"] < mwds_data["cost"]: winner_cost = AUTHOR_SOLVER_NAME
            elif mwds_data["cost"] < author_data["cost"]: winner_cost = MWDS_DEEPOPT_SOLVER_NAME
            else:
                winner_cost = "Tie"
                t_auth, t_mwds = author_data.get("time"), mwds_data.get("time")
                if t_auth is not None and t_mwds is not None:
                    if t_auth < t_mwds: winner_time = AUTHOR_SOLVER_NAME
                    elif t_mwds < t_auth: winner_time = MWDS_DEEPOPT_SOLVER_NAME
                    else: winner_time = "Tie"
                elif t_auth is not None: winner_time = AUTHOR_SOLVER_NAME # One has time, the other doesn't
                elif t_mwds is not None: winner_time = MWDS_DEEPOPT_SOLVER_NAME
        elif author_data: winner_cost = winner_time = AUTHOR_SOLVER_NAME
        elif mwds_data: winner_cost = winner_time = MWDS_DEEPOPT_SOLVER_NAME
        summary[graph]["winner_by_cost"] = winner_cost
        summary[graph]["winner_by_time_if_cost_tied_or_single"] = winner_time
    return summary

def find_best_overall_results(all_run_data):
    best_results = {}
    program_names = [AUTHOR_SOLVER_NAME, MWDS_DEEPOPT_SOLVER_NAME]
    for prog_name in program_names:
        best_results[prog_name] = {
            "best_cost": float('inf'), "graph_file": None, "seed_used": None,
            "time_for_best_cost_reported_sec": None, "script_wall_time_for_best_cost_sec": None,
            "solution_size_for_best_cost": None, "round_number": None, "status_of_best_run": None,
            "valgrind_errors_at_best": None,
            "valgrind_log_at_best": None,
            "is_feasible_at_best": None # Carry feasibility along
        }

    for run in all_run_data:
        valgrind_metrics = run.get("valgrind_metrics")
        run_had_valgrind_errors = valgrind_metrics.get("valgrind_errors_found", False) if valgrind_metrics else None
        
        if run["status"] == "completed" and run["solution_cost"] is not None:
            prog_name = run["program_name"]
            current_best_cost = best_results[prog_name]["best_cost"]
            new_cost = run["solution_cost"]
            update = False
            current_best_had_valgrind_errors = best_results[prog_name]["valgrind_errors_at_best"]

            if new_cost < current_best_cost:
                update = True
            elif new_cost == current_best_cost:
                # Prioritize runs without valgrind errors if current best has them
                if current_best_had_valgrind_errors is True and run_had_valgrind_errors is False:
                    update = True
                elif current_best_had_valgrind_errors == run_had_valgrind_errors: # Or if both have same valgrind error status
                    time_fields_current = (best_results[prog_name]["time_for_best_cost_reported_sec"], best_results[prog_name]["script_wall_time_for_best_cost_sec"])
                    time_fields_new = (run["program_time_reported_sec"], run["script_wall_time_sec"])
                    
                    current_time_to_compare = time_fields_current[0] if time_fields_current[0] is not None else time_fields_current[1]
                    new_time_to_compare = time_fields_new[0] if time_fields_new[0] is not None else time_fields_new[1]

                    if current_time_to_compare is None or \
                       (new_time_to_compare is not None and new_time_to_compare < current_time_to_compare):
                        update = True
            
            if update:
                best_results[prog_name].update({
                    "best_cost": new_cost, "graph_file": run["graph_file"], "seed_used": run["seed_used"],
                    "time_for_best_cost_reported_sec": run["program_time_reported_sec"],
                    "script_wall_time_for_best_cost_sec": run["script_wall_time_sec"],
                    "solution_size_for_best_cost": run["solution_size"],
                    "round_number": run["round_number"], "status_of_best_run": run["status"],
                    "valgrind_errors_at_best": run_had_valgrind_errors if run.get("use_valgrind") else None,
                    "valgrind_log_at_best": run.get("valgrind_log_file") if run_had_valgrind_errors else None,
                    "is_feasible_at_best": run.get("is_feasible") # Store feasibility of the best run
                })

    for prog_name in program_names:
        if best_results[prog_name]["best_cost"] == float('inf'):
            best_results[prog_name]["best_cost"] = "N/A (No 'completed' runs with cost)"
    return best_results

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ MWDS solvers with optional Valgrind and systemd-run.")
    # MODIFIED: Updated help string for graph_folder
    parser.add_argument("graph_folder", help="Path pattern to .clq graph files (e.g., 'folder/*.clq', 'specific_folder/myprefix_*.clq', or 'abc/ddd*.clq').")
    parser.add_argument("output_json", help="JSON file to save results and comparison.")
    parser.add_argument("time_limit", type=int, help="Base time limit in seconds for each solver program run. Increase significantly if using --use-valgrind.")
    parser.add_argument("--max_concurrent_runs", type=int, default=0, help="Max concurrent runs. 0 for #logical_cores.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of benchmark rounds.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for solvers. If None, random per run. If set, solvers use base_seed + round_offset.")
    parser.add_argument("--compiler_choice", choices=['gcc', 'llvm'], default='gcc', help="Compiler: gcc or llvm. Default: gcc")
    parser.add_argument("--use-systemd-run", action="store_true", help="[Linux only] Wrap solver with systemd-run for resource isolation.")
    parser.add_argument("--solver-memory-limit-mb", type=int, default=1024, help="[Used with --use-systemd-run] MemoryMax in MB for the solver's cgroup.")
    parser.add_argument("--use-valgrind", action="store_true", help="[Linux only] Wrap solver with Valgrind to detect memory leaks/errors. WARNING: Significantly increases runtime; adjust time_limit accordingly.")
    parser.add_argument("--valgrind-log-dir", type=str, default="valgrind_logs", help="Directory to store Valgrind log files.")
    args = parser.parse_args()

    if args.use_systemd_run and not is_tool_available("systemd-run"):
        print("Error: --use-systemd-run specified, but 'systemd-run' command not found.")
        return
    if args.use_valgrind and not is_tool_available("valgrind"):
        print("Error: --use-valgrind specified, but 'valgrind' command not found.")
        return
    if not args.use_systemd_run and not is_tool_available("taskset"):
        print("Warning: 'taskset' not found and --use-systemd-run is not active. Core pinning disabled.")

    build_dir = Path("build_bench"); build_dir.mkdir(exist_ok=True)
    executables, compilation_logs = {}, {}
    author_source, mwds_source = "author.cpp", "mwds-deepopt.cpp" # Assuming my.cpp is named mwds-deepopt.cpp on disk
    selected_compiler_binary = COMPILER_BINARIES.get(args.compiler_choice)
    if not selected_compiler_binary: print(f"Error: Unknown compiler '{args.compiler_choice}'."); return
    print(f"Using compiler: {args.compiler_choice} ({selected_compiler_binary})")
    if not Path(author_source).exists(): print(f"Error: {author_source} not found."); return
    if not Path(mwds_source).exists(): print(f"Error: {mwds_source} not found (expected for MWDS_DEEPOPT_SOLVER_NAME)."); return
    
    for name, src, flags in [(AUTHOR_SOLVER_NAME, author_source, BASE_COMPILE_FLAGS[AUTHOR_SOLVER_NAME]),
                             (MWDS_DEEPOPT_SOLVER_NAME, mwds_source, BASE_COMPILE_FLAGS[MWDS_DEEPOPT_SOLVER_NAME])]:
        succ, log = compile_cpp(src, str(build_dir / name), selected_compiler_binary, flags)
        compilation_logs[name] = {"compiler_used": selected_compiler_binary, "log": log}
        if not succ: print(f"Compilation failed for {name}. Exiting."); return
        executables[name] = str(Path(build_dir / name).resolve())

    # MODIFIED: Use args.graph_folder directly as the glob pattern
    graph_files_raw = glob.glob(args.graph_folder)
    if not graph_files_raw:
        # MODIFIED: Update error message to reflect pattern usage
        print(f"No .clq files found matching the pattern: {args.graph_folder}"); return
    graph_files = [str(Path(g).resolve()) for g in graph_files_raw]
    print(f"Found {len(graph_files)} graph files (paths resolved to absolute).")

    if args.use_valgrind:
        valgrind_log_dir_path = Path(args.valgrind_log_dir)
        valgrind_log_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Valgrind logs will be stored in: {valgrind_log_dir_path.resolve()}")
        print("WARNING: Valgrind is enabled. Runtimes will be much longer. Ensure 'time_limit' is appropriately high.")
    if args.use_systemd_run:
        print(f"INFO: systemd-run will be used with MemoryMax={args.solver_memory_limit_mb}MB and CPUAffinity.")

    global_tasks_to_run = []
    seed_generation_strategy = ""
    if args.seed is None:
        seed_generation_strategy = "Independently randomized per run for each solver."
        print(f"INFO: Seeds for solvers: {seed_generation_strategy}")
    else:
        seed_generation_strategy = f"Deterministic: base_seed ({args.seed}) + round_offset, same for both solvers in a given round."
        print(f"INFO: Seeds for solvers: {seed_generation_strategy}")

    for r_idx in range(args.rounds):
        for g_file_abs_path in graph_files:
            for p_name in [AUTHOR_SOLVER_NAME, MWDS_DEEPOPT_SOLVER_NAME]:
                seed_for_this_task = 0
                if args.seed is not None:
                    seed_for_this_task = args.seed + r_idx
                else:
                    seed_for_this_task = random.randint(0, 2**32 - 1)
                
                task = {"program_name": p_name,
                        "executable_path": executables[p_name],
                        "graph_file": g_file_abs_path,
                        "time_limit_seconds": args.time_limit,
                        "round_number": r_idx,
                        "seed_to_use": seed_for_this_task, # Centralized seed for the task
                        "use_systemd_run": args.use_systemd_run,
                        "solver_memory_limit_mb": args.solver_memory_limit_mb,
                        "use_valgrind": args.use_valgrind,
                        "valgrind_log_dir": args.valgrind_log_dir
                        }
                global_tasks_to_run.append(task)
    
    if not global_tasks_to_run: print("No tasks generated. Exiting."); return
    print(f"Generated {len(global_tasks_to_run)} tasks across {args.rounds} round(s).")

    all_results_data, result_q = [], Queue()
    num_logical_cores, num_physical_cores = get_cpu_counts()
    smt_active = num_physical_cores > 0 and num_logical_cores > num_physical_cores

    if smt_active:
        print(f"SMT Detected: {num_physical_cores} physical, {num_logical_cores} logical cores. Prioritizing primary logical cores.")
        available_primary_cores = sorted(list(range(num_physical_cores)))
        available_smt_cores = sorted(list(range(num_physical_cores, num_logical_cores)))
    else:
        print(f"No SMT detected or SMT factor is 1 (or could not reliably determine physical cores): {num_physical_cores} physical (reported), {num_logical_cores} logical cores.")
        available_primary_cores = sorted(list(range(num_logical_cores)))
        available_smt_cores = []

    actual_max_procs = args.max_concurrent_runs if args.max_concurrent_runs > 0 else num_logical_cores
    print(f"Using up to {actual_max_procs} concurrent processes.")
    print(f"Solver time limit reduction: {PROGRAM_INTERNAL_TIME_LIMIT_REDUCTION_SEC}s. Script subprocess timeout factor: {SCRIPT_SUBPROCESS_TIMEOUT_FACTOR}x.")
    print(f"Launch Gaps: Short (<{LOW_MEMORY_THRESHOLD_PERCENT}% mem): {SHORT_LAUNCH_GAP_SECONDS}s, Long (>= {HIGH_MEMORY_THRESHOLD_PERCENT}% mem): {LONG_LAUNCH_GAP_SECONDS}s")
    print(f"Max memory for new launches: {MAX_MEMORY_CEILING_PERCENT}%. If mem >= {HIGH_MEMORY_THRESHOLD_PERCENT}% post-gap, wait {POST_GAP_HIGH_MEM_WAIT_SECONDS}s.")

    active_procs, last_launch_t = [], 0
    total_gen_tasks, completed_tasks = len(global_tasks_to_run), 0
    tasks_to_do = list(global_tasks_to_run)
    print(f"Starting benchmark execution for {total_gen_tasks} tasks...")

    while tasks_to_do or active_procs:
        for i in range(len(active_procs) - 1, -1, -1):
            proc, core_id = active_procs[i]
            if not proc.is_alive():
                proc.join(); active_procs.pop(i)
                if smt_active:
                    if 0 <= core_id < num_physical_cores: available_primary_cores.append(core_id); available_primary_cores.sort()
                    else: available_smt_cores.append(core_id); available_smt_cores.sort()
                else:
                    available_primary_cores.append(core_id); available_primary_cores.sort()
        
        newly_done = 0
        while not result_q.empty():
            all_results_data.append(result_q.get())
            newly_done +=1
        if newly_done > 0:
            completed_tasks += newly_done
            print(f"Progress: {completed_tasks}/{total_gen_tasks} tasks completed. Active: {len(active_procs)}")

        mem_usage = psutil.virtual_memory().percent
        current_gap = SHORT_LAUNCH_GAP_SECONDS
        if mem_usage >= HIGH_MEMORY_THRESHOLD_PERCENT: current_gap = LONG_LAUNCH_GAP_SECONDS
        # No need for specific LOW_MEMORY_THRESHOLD_PERCENT check if SHORT is default
        
        time_since_launch = time.time() - last_launch_t
        gap_ok = (time_since_launch >= current_gap or not active_procs) # Simplified: if no active procs, gap is fine.
        can_launch = False

        if tasks_to_do and len(active_procs) < actual_max_procs and gap_ok:
            mem_after_gap = psutil.virtual_memory().percent
            if mem_after_gap >= HIGH_MEMORY_THRESHOLD_PERCENT and active_procs: # Only wait if there are active processes consuming memory
                print(f"R*| Mem ({mem_after_gap:.1f}%) still high (>{HIGH_MEMORY_THRESHOLD_PERCENT}%) after {current_gap}s gap. Waiting {POST_GAP_HIGH_MEM_WAIT_SECONDS}s...")
                time.sleep(POST_GAP_HIGH_MEM_WAIT_SECONDS)
                # Re-check memory after waiting, for the MAX_MEMORY_CEILING_PERCENT condition
                mem_after_gap = psutil.virtual_memory().percent 
            
            if mem_after_gap < MAX_MEMORY_CEILING_PERCENT:
                if smt_active:
                    if available_primary_cores or available_smt_cores: can_launch = True
                else:
                    if available_primary_cores: can_launch = True
            else: print(f"R*| Mem ({mem_after_gap:.1f}%) > ceiling ({MAX_MEMORY_CEILING_PERCENT}%). Holding launch.")

        if can_launch:
            task_spec = tasks_to_do.pop(0)
            core_assign = None
            if smt_active:
                if available_primary_cores: core_assign = available_primary_cores.pop(0)
                elif available_smt_cores: core_assign = available_smt_cores.pop(0)
            elif available_primary_cores:
                 core_assign = available_primary_cores.pop(0)

            if core_assign is not None:
                task_spec["core_id"] = core_assign
                p = Process(target=run_benchmark_task, args=(task_spec, result_q))
                p.start(); active_procs.append((p, core_assign)); last_launch_t = time.time()
            elif tasks_to_do : # Check if task was popped but no core assigned
                 tasks_to_do.insert(0, task_spec); # Requeue if no core available
                 print(f"Warning: Launch intended for {task_spec['program_name']} but no core assigned (all {actual_max_procs} busy or no available types). Task requeued.")
        
        if tasks_to_do or active_procs: time.sleep(0.5) # Main loop sleep
        else: break # Exit loop if no tasks left and no active processes

    while not result_q.empty(): all_results_data.append(result_q.get()); completed_tasks +=1 # Final drain
    if completed_tasks < total_gen_tasks and total_gen_tasks > 0 : # Ensure total_gen_tasks is not zero
        print(f"Final Progress: {completed_tasks}/{total_gen_tasks} tasks completed.")
    print(f"\nAll {completed_tasks} launched tasks concluded.")

    results_by_round = {}
    for item in all_results_data:
        r_num = item.get("round_number", -1) # Default to -1 if missing
        try: r_num = int(r_num)
        except (ValueError, TypeError): print(f"Warn: Invalid round_num '{r_num}'. Assigning -1."); r_num = -1
        results_by_round.setdefault(r_num, []).append(item)

    summaries_per_round = {}
    for r_idx_key in sorted(results_by_round.keys()):
        if r_idx_key == -1: print(f"Warn: Skipping summary for results with invalid round_number (-1)."); continue
        print(f"Generating comparison summary for round {r_idx_key +1}...") # Assuming r_idx_key is 0-indexed
        summaries_per_round[f"round_{r_idx_key}"] = generate_comparison_summary(results_by_round[r_idx_key])

    best_overall = find_best_overall_results(all_results_data)
    print("Generated best overall results per program.")

    # MODIFIED: Use args.graph_folder directly for the parameter logging
    final_json_parameters = {
        "graph_folder_pattern": args.graph_folder, # Changed key name for clarity
        "output_json_path": str(Path(args.output_json).resolve()),
        "time_limit_per_solver_run_base_sec": args.time_limit,
        "program_internal_time_limit_reduction_sec": PROGRAM_INTERNAL_TIME_LIMIT_REDUCTION_SEC,
        "script_subprocess_timeout_factor": SCRIPT_SUBPROCESS_TIMEOUT_FACTOR,
        "max_concurrent_runs_setting": args.max_concurrent_runs, "actual_max_procs_used": actual_max_procs,
        "compiler_choice": args.compiler_choice, "selected_compiler_binary": selected_compiler_binary,
        "cpu_info": {"logical_cores": num_logical_cores, "physical_cores": num_physical_cores, "smt_active": smt_active},
        "linux_tool_options": {
            "use_systemd_run": args.use_systemd_run,
            "solver_memory_limit_mb": args.solver_memory_limit_mb if args.use_systemd_run else None,
            "use_valgrind": args.use_valgrind,
            "valgrind_log_dir": str(Path(args.valgrind_log_dir).resolve()) if args.use_valgrind else None,
            "valgrind_error_report_code": VALGRIND_ERROR_REPORT_CODE if args.use_valgrind else None,
        },
        "memory_and_launch_gaps": {
            "low_mem_threshold_percent": LOW_MEMORY_THRESHOLD_PERCENT,
            "high_mem_threshold_percent_for_long_gap": HIGH_MEMORY_THRESHOLD_PERCENT,
            "short_launch_gap_seconds": SHORT_LAUNCH_GAP_SECONDS,
            "long_launch_gap_seconds": LONG_LAUNCH_GAP_SECONDS,
            "max_memory_ceiling_for_launch_percent": MAX_MEMORY_CEILING_PERCENT,
            "post_gap_high_mem_wait_seconds": POST_GAP_HIGH_MEM_WAIT_SECONDS
        },
        "cli_seed_base": args.seed, # Renamed for clarity
        "seed_generation_strategy": seed_generation_strategy,
        "num_rounds_configured": args.rounds,
    }

    final_json_output = { # Renamed to avoid conflict with json module
        "parameters": final_json_parameters,
        "compilation_logs": compilation_logs,
        "runs": sorted(all_results_data, key=lambda x: (x.get("round_number", 0), str(x.get("graph_file")), str(x.get("program_name")))),
        "comparison_summary_per_round": summaries_per_round,
        "best_overall_results_per_program": best_overall
    }
    try:
        with open(args.output_json, 'w') as f: json.dump(final_json_output, f, indent=4)
        print(f"Results saved to {args.output_json}")
    except IOError as e: print(f"Error writing results to {args.output_json}: {e}")
    print("Benchmark finished.")

if __name__ == "__main__":
    main()