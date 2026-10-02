#!/usr/bin/env python3
"""Live SM86 development probes: native correctness -> trace -> predict -> NCU.

All case predictions are saved before the first profiler capture. Repetitions
are separate application-replay processes. No observation is passed to sim-run;
these fixed cases are development experiments, never fresh held-out accuracy.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
CUDA = Path('/usr/local/cuda-13.2')
METRICS = [
    'smsp__inst_executed.sum', 'smsp__inst_executed_pipe_alu.sum',
    'smsp__inst_executed_pipe_fma.sum', 'smsp__inst_executed_pipe_lsu.sum',
    'gpc__cycles_elapsed.max', 'gpu__time_duration.sum', 'sm__cycles_active.avg',
    'dram__sectors_read.sum', 'dram__sectors_write.sum', 'lts__t_sectors_srcunit_tex.sum',
] + [f'l1tex__t_{quantity}_pipe_lsu_mem_global_op_{op}.sum'
     for op in ('ld', 'st', 'atom', 'red') for quantity in ('requests', 'sectors')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run(command, directory, label, env=None, timeout=240):
    """Bound each process group; retain failures and actual child peak RSS."""
    directory.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    (directory / f'{label}.command.txt').write_text(shlex.join(command) + '\n')
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    timed_out = False
    with (directory / f'{label}.stdout').open('w') as out, (directory / f'{label}.stderr').open('w') as err:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=out,
                                   stderr=err, start_new_session=True)
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() - start > timeout and not timed_out:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.005)
    result = {
        'exit_code': process.returncode, 'timed_out': timed_out,
        'started_at': started_at, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - start,
        'peak_rss_bytes': usage.ru_maxrss * 1024,
        'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
    }
    write_json(directory / f'{label}.result.json', result)
    if process.returncode or timed_out:
        raise RuntimeError(f'{label} failed: {result}; {(directory / f"{label}.stderr").read_text()[-1200:]}')
    return result


def clean_environment():
    env = dict(os.environ)
    for key in list(env):
        if key in ('CUDA_INJECTION64_PATH', 'LD_PRELOAD', 'NVBIT_TOOL') or key.startswith('SIM_SM_REFERENCE_'):
            env.pop(key)
    return env


def trace_environment(directory, base):
    env = dict(base)
    env.update({
        'CUDA_INJECTION64_PATH': str(REPO / 'tools/nvbit_tracer/tracer_tool/tracer_tool.so'),
        'NVBIT_ALPHA_NO_HASH': '1',
        'LD_LIBRARY_PATH': str(REPO / '.toolchain/pb/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
        'PATH': str(CUDA / 'bin') + ':' + env['PATH'],
        'TRACES_FOLDER': str(directory / 'trace'), 'USER_DEFINED_FOLDERS': '1',
        'ACTIVE_FROM_START': '1', 'DYNAMIC_KERNEL_LIMIT_START': '0', 'DYNAMIC_KERNEL_LIMIT_END': '1',
        'SKIP_ENHANCED_TRACER': '0', 'ENHANCED_TRACER_MAX_PARSER_INPUT_KB': '1024',
    })
    write_json(directory / 'trace.environment.json', {k: v for k, v in env.items() if base.get(k) != v})
    return env


def capture_prediction(directory, command, control, sim_run, profile, base):
    run(command, directory, 'native', base)
    run(command, directory, 'trace', trace_environment(directory, base))
    run([sim_run, '--trace', directory / 'trace', '--control', control,
         '--arch-card', REPO / 'arch_profiles/cards/sm_86.json',
         '--pchase-card', REPO / 'data/pchase/sm_86_static_card.json',
         '--mode', 'l1-l2', '--cache-scope', 'launch', '--latent-envelope',
         '--predictive-service-profile', profile], directory, 'replay',
        {**base, 'SIM_RUN_PERFORMANCE_COUNTERS': '1'})
    # Analyzer execution is measured separately; it is not native acceleration.
    run([sim_run, '--trace', directory / 'trace', '--control', control,
         '--arch-card', REPO / 'arch_profiles/cards/sm_86.json',
         '--pchase-card', REPO / 'data/pchase/sm_86_static_card.json',
         '--fast-frontier-only'], directory, 'frontier', base)


def parse_observation(path):
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        raise ValueError(f'{path}: missing NCU header')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    units = next(reader, {})
    rows = [row for row in reader if (row.get('ID') or '').isdigit()]
    if len(rows) != 1:
        raise ValueError(f'{path}: expected one measured launch, found {len(rows)}')
    row = rows[0]
    values = {}
    for metric in METRICS:
        raw = row.get(metric)
        if raw in (None, '', 'N/A'):
            values[metric] = None
            continue
        value = float(raw.replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{path}: invalid {metric}={raw}')
        values[metric] = value
    target = 'gpc__cycles_elapsed.max'
    if values[target] is None or units.get(target) != 'cycle':
        raise ValueError(f'{path}: missing target cycles or incorrect unit')
    return {'values': values, 'units': {m: units.get(m) for m in METRICS},
            'profiler_fields': {k: row.get(k) for k in ('ID', 'Kernel Name', 'Device', 'CC', 'Grid Size', 'Block Size', 'Context', 'Stream')}}


def collect_observations(directory, command, repetitions, base):
    rows = []
    for repetition in range(repetitions):
        output = directory / f'ncu_{repetition}'
        run([REPO / '.venv/bin/python', REPO / 'tools/local_labels/collect_ncu.py',
             '--output', output, '--ncu', CUDA / 'nsight-compute-2026.1.0/ncu',
             '--timeout', '180', '--launch-count', '1', '--replay-mode', 'application',
             '--clock-control', 'none', '--cache-control', 'all',
             '--metrics', ','.join(METRICS), '--', *command], directory, f'ncu_{repetition}', base)
        rows.append(parse_observation(output / 'ncu.csv'))
    write_json(directory / 'observations.json', rows)
    return rows


def summarize_case(directory, observations):
    replay = json.loads((directory / 'replay.stdout').read_text())
    launch = replay['launches'][0]
    projection = replay['latent_envelope']['launches'][0]['ncu_projection']
    values = [o['values']['gpc__cycles_elapsed.max'] for o in observations]
    target = statistics.median(values)
    lower, upper = projection['predicted_lower_cycles'], projection['predicted_upper_cycles']
    domain_checks = {'instructions': all(o['values']['smsp__inst_executed.sum'] == launch['counters']['instructions_issued'] for o in observations)}
    metrics = launch['native_metrics']['metrics']
    for short, name in [('ld', 'load'), ('st', 'store'), ('atom', 'atomic'), ('red', 'reduction')]:
        for quantity in ['requests', 'sectors']:
            key = f'global_{name}_{quantity}'
            expected = metrics[key]['value']
            domain_checks[key] = all(o['values'][f'l1tex__t_{quantity}_pipe_lsu_mem_global_op_{short}.sum'] == expected for o in observations)
    frontier = json.loads((directory / 'frontier.stdout').read_text())
    return {
        **json.loads((directory / 'case.json').read_text()), 'status': 'completed',
        'instructions': launch['counters']['instructions_issued'],
        'terminal_modeled_event': launch['timing']['terminal_modeled_event_time'],
        'observer_candidate_lower_cycles': lower, 'observer_candidate_upper_cycles': upper,
        'ncu_cycles_samples': values, 'ncu_cycles_median': target,
        'ncu_range_over_median_pct': (max(values) - min(values)) / target * 100 if target else None,
        'lower_point_ape_pct': abs(lower - target) / target * 100 if lower is not None and target else None,
        'finite_interval_covered': lower <= target <= upper if lower is not None and upper is not None else None,
        'open_lower_consistent': lower <= target if lower is not None else None,
        'domain_checks_all_repetitions': domain_checks,
        'trace_metrics': {k: v for k, v in metrics.items() if k in ('dram_read_bytes', 'dirty_eviction_sectors', 'terminal_dirty_sectors', 'dram_write_sectors')},
        'ncu_dram_read_sectors': [o['values']['dram__sectors_read.sum'] for o in observations],
        'ncu_dram_write_sectors': [o['values']['dram__sectors_write.sum'] for o in observations],
        'completion': launch['timing']['completion_frontier'],
        'replay_process_cost': json.loads((directory / 'replay.result.json').read_text()),
        'analyzer_process_cost': json.loads((directory / 'frontier.result.json').read_text()),
        'analyzer_coverage': [r['summary']['coverage'] for r in frontier.get('fast_frontier', [])],
        'analyzer_macro_dispatches': sum(r['summary'].get('macro_dispatches', 0) for r in frontier.get('fast_frontier', [])),
        'native_macro_execution': 'not_enabled',
    }


def execute_cases(cases, root, control, sim_run, profile, repetitions, base):
    predicted = []
    results = []
    # Deliberately finish the ENTIRE prediction pass before collecting scores.
    for directory, command in cases:
        try:
            capture_prediction(directory, command, control, sim_run, profile, base)
            predicted.append((directory, command))
            print(f'predicted {directory.name}', flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            results.append({'case': directory.name, 'status': 'prediction_failed', 'error': str(error)})
            print(f'FAILED prediction {directory.name}: {error}', flush=True)
    write_json(root / 'prediction_phase_complete.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'predicted_cases': [directory.name for directory, _ in predicted],
        'failed_cases': [row['case'] for row in results if row['status'] == 'prediction_failed'],
        'ncu_started': False,
    })
    for directory, command in predicted:
        try:
            observations = collect_observations(directory, command, repetitions, base)
            results.append(summarize_case(directory, observations))
            print(f'measured {directory.name} x{repetitions}', flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            results.append({'case': directory.name, 'status': 'observation_failed', 'error': str(error)})
            print(f'FAILED observation {directory.name}: {error}', flush=True)
        write_json(root / 'summary.json', {'role': 'live_development_no_fitting', 'rows': results})
    write_json(root / 'observation_phase_complete.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'observed_cases': [row['case'] for row in results if row['status'] == 'completed'],
        'prediction_phase_completed_first': True,
        'repetitions_per_case': repetitions,
    })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', choices=('small', 'large'), default='small')
    parser.add_argument('--suite', choices=('domains', 'writeback'), default='domains')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--sim-run', type=Path, default=REPO / 'sim/target/release/sim-run')
    parser.add_argument('--profile', type=Path, default=REPO / 'docs/sm86_service_rate_profile_20261001.json')
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error('--repetitions must be 1..20')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('use a fresh output directory; previous experiments are not overwritten')
    root.mkdir(parents=True, exist_ok=True)
    base = clean_environment()
    host_command = ['nvidia-smi', '--query-gpu=index,name,uuid,compute_cap,driver_version,pstate,clocks.current.sm,clocks.current.memory', '--format=csv']
    run(host_command, root, 'host_before', base)
    run([CUDA / 'bin/nvcc', '--version'], root, 'nvcc_version', base)
    run([CUDA / 'nsight-compute-2026.1.0/ncu', '--version'], root, 'ncu_version', base)
    binary = root / 'metric_domains'
    source = Path(__file__).with_name('main_writeback.cu' if args.suite == 'writeback' else 'main.cu')
    run([CUDA / 'bin/nvcc', '-ccbin=/usr/bin/g++-14', '-arch=sm_86', '-O2', '-lineinfo', source, '-o', binary], root, 'build', base)
    control = root / 'control.json'
    run([REPO / '.venv/bin/python', REPO / 'tools/sass_control/emit_control_bits_sidecar.py', '--binary', binary,
         '--arch', 'sm_86', '--toolkit-bindir', CUDA / 'bin', '--out', control], root, 'control', base)
    blocks, iterations = (1, 8) if args.size == 'small' else (168, 32)
    if args.suite == 'writeback':
        shapes = ([('store', 0, 1, 8), ('store', 4, 1, 8), ('control', 0, 1, 8)] if args.size == 'small' else
                  [('store', 4, 256, 128), ('store', 4, 1024, 128)])
    else:
        shapes = [(op, p, blocks, iterations) for op in ('read', 'atom', 'red')
                  for p in ((0, 1, 2) if args.size == 'small' else (0, 2, 3))]
        shapes += [('control', 0, blocks, iterations)]
    cases = []
    for op, pattern, blocks, iterations in shapes:
        name = f'{args.size}_{op}_{pattern}' + (f'_b{blocks}_i{iterations}' if args.suite == 'writeback' else '')
        directory = root / name
        directory.mkdir()
        write_json(directory / 'case.json', {'case': name, 'operation': op, 'pattern': pattern,
                   'blocks': blocks, 'iterations': iterations, 'suite': args.suite})
        cases.append((directory, [binary, op, pattern, blocks, iterations]))
    write_json(root / 'inputs.json', {
        'sim_run': str(args.sim_run.resolve()), 'profile': str(args.profile.resolve()),
        'source': str(source), 'binary': str(binary), 'control': str(control),
        'arch_card': str(REPO / 'arch_profiles/cards/sm_86.json'),
        'pchase_card': str(REPO / 'data/pchase/sm_86_static_card.json'),
        'repetitions': args.repetitions, 'cache_control': 'all', 'clock_control': 'none',
        'cuda_visible_devices': base.get('CUDA_VISIBLE_DEVICES'),
        'role': 'development_not_holdout_no_parameter_fitting',
    })
    results = execute_cases(cases, root, control, args.sim_run.resolve(), args.profile.resolve(), args.repetitions, base)
    run(host_command, root, 'host_after', base)
    write_json(root / 'summary.json', {'role': 'live_development_no_fitting', 'rows': results})
    return int(any(row['status'] != 'completed' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
