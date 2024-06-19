from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator, FourPeaks, FlipFlopGenerator, KnapsackGenerator
import mlrose_hiive
from mlrose_hiive import SARunner, GARunner, NNGSRunner
import numpy as np
import pandas as pd


def genaerate_problems_over_size(length = [ 10, 20, 40, 80 , 160 ]):
    """_summary_

    Args:
        length (list, optional): _description_. Defaults to [ 10, 20, 40, 80 , 160 ].
        
    Returns:
        list (tuple): list of tuples with the length and the problem object.
        
    
    """
    t_pct = 0.2
    fitness = FourPeaks(t_pct=t_pct)
    length = [ 10, 20, 40, 80 , 160 ]
    problems = []
    for l in length:
        problem = DiscreteOpt(length=l, fitness_fn=fitness,  maximize = True, max_val = 2)

        maximum_expected = l - (t_pct* l + 1) +l
        print(f"maximum_expected: {maximum_expected} for size {l}")
        problems.append((l,problem))
    
    return problems
    
    
    
def Runner_overSeeds( problem, 
                        runner_algorthm = SARunner,
                        seeds = [1,2,3,4,5],
                        **kwargs
                        ):
    """_summary_

    Args:
        problem (tuple): _description_
        seeds (list, optional): _description_. Defaults to [1,2,3,4,5].
        max_attempts (int, optional): _description_. Defaults to 10.
        max_iters (int, optional): _description_. Defaults to 1000.
        init_state ([type], optional): _description_. Defaults to None.
        decay (float, optional): _description_. Defaults to 0.99.
        schedule ([type], optional): _description_. Defaults to mlrose_hiive.GeomDecay().
        
    Returns:
        list (tuple): list of tuples with the seed and the best state found.
        
    """
    df_run_stats_total = []
    df_run_curves_total = []
    for seed in seeds:
        np.random.seed(seed)
        
        runner = runner_algorthm(problem = problem, seed = seed, **kwargs)
            # the two data frames will contain the results
        df_run_stats, df_run_curves = runner.run()
        df_run_stats["seed"] = seed
        df_run_curves["seed"] = seed
        df_run_stats_total.append(df_run_stats)
        df_run_curves_total.append(df_run_curves)
        
        # print(best_fitness)
        # 2.0

    # concat the dataframes
    df_run_stats_total = pd.concat(df_run_stats_total)
    df_run_curves_total = pd.concat(df_run_curves_total)
    
    # save four peaks total stats to disk
    df_run_stats_total.to_csv("four_peaks_total_stats.csv")
    df_run_curves_total.to_csv("four_peaks_total_curves.csv")
    
    return df_run_stats_total, df_run_curves_total