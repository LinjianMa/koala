from experiments.tfi_ite_peps import TraversalFieldIsing, build_cli_parser, run_peps
from experiments.benchmark import Benchmark
from koala import peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD



def run(args):
    include = (Snake, BMPS, ABMPS, TRG, Square)
    exclude = ()
    tfi = TraversalFieldIsing(args.coupling, args.field, args.nrow, args.ncol, args.tau, args.backend)
    qstate = run_peps(tfi, args.steps, args.normfreq, backend=args.backend, maxrank=args.maxrank)
    maxrank = args.maxrank # ** 2
    standard = None

    options = []
    for contract_option in include:
        if contract_option not in exclude:
            if contract_option is Snake:
                options.append(Snake())
            else:
                for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                    options.append(TRG(svd_option, svd_option) if contract_option is TRG else contract_option(svd_option))
    
    for option in options:
        bm = Benchmark(str(option), qstate.backend, standard=standard, path=args.path, 
            reps=1, profile_time=args.profile_time, profile_memory=args.profile_memory, additional_info={
                'coupling': args.coupling,
                'field': args.field,
                'tau': args.tau,
                'steps': args.steps,
                'normfreq': args.normfreq,
            })
        bm.add_PEPS_info(qstate)
        bm.add_contract_info(option)
        with bm:
            bm.result = qstate.expectation(tfi.observable, contract_option=option)
            
        if isinstance(option, Snake) and not standard:
            standard = bm.result



if __name__ == '__main__':
    parser = build_cli_parser()
    parser.add_argument('-p', '--path', help='path to save the data file', type=str, default=None)
    parser.add_argument('-pt', '--profile-time', dest='profile_time', action='store_const', const='True', 
        default=False, help='enable profiling time')
    parser.add_argument('-pm', '--profile-memory', dest='profile_memory', action='store_const', const='True', 
        default=False, help='enable profiling memory')

    run(parser.parse_args())
