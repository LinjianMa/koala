from time import process_time
import subprocess


def benchmark_norm(p, contract_option, standard=1, path=None, verbosity=1, reps=1, profile=True):
    backend = p.backend
    is_ctf = backend.name in {'ctf', 'ctfview'}
    if profile:
        if is_ctf:
            import ctf
            timer = ctf.timer_epoch(f'{contract_option}')
            timer.begin()
        else:
            import cProfile
            import pstats
            from io import StringIO
            
            pr = cProfile.Profile()
            pr.enable()

    start = process_time()
    best = None
    for i in range(reps):
        loop_start = process_time()
        result = p.norm(contract_option)
        loop_duration = process_time() - loop_start
        if not best or best > loop_duration:
            best = loop_duration
    duration = process_time() - start
    
    if profile:
        if is_ctf:
            timer.end()
        else:
            pr.disable()

    abs_err = abs(result - standard)
    rel_err = abs(abs_err / standard)

    data = {
        'shape': p.shape,
        'bond_dim': p[0,0].shape[1],
        'contract_option': str(contract_option),
        'result': result,
        'time': duration / reps,
        'abs_err': abs_err,
        'rel_err': rel_err,
        'backend': backend.name,
        'nproc': backend.nproc,
        'git_hash': subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    }

    if backend.rank == 0:
        if verbosity > 0:
            print(f'Contract option: {contract_option}')
            print(f'Total time spent for {reps} repetitions: {duration}')
            print('Average time per loop {0}'.format(duration / reps))
            if verbosity > 1:
                print(f'Best loop {best}')
                print(f'Result: {result} Standard: {standard}')
            print(f'Relative error: {rel_err} Absolute error: {abs_err}')

            if profile and not is_ctf:
                s = StringIO()
                ps = pstats.Stats(pr, stream=s)
                ps.sort_stats('cumtime').print_stats(15)
                ps.sort_stats('tottime').print_stats(15)
                print(s.getvalue())
                
        if path:
            import json
            with open(path, 'r') as fd:
                database = json.load(fd)
            with open(path, 'w+') as fd:
                # database[str(contract_option)] = database.get(str(contract_option), []).append(data)
                database.append(data)
                json.dump(database, fd, indent=4)

    return data

