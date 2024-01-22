import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

import numpy as np
import copy
from evaluation.critical import *
from experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
import logging as log
from model_ga.population import PopulationExtended
from model_ga.result import ResultExtended
import problem
import os
from problem.pymoo_test_problem import PymooTestProblem
from utils.evaluation import evaluate_individuals
from visualization import output
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from utils.sorting import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import random
from pymoo.termination import get_termination
import pyswarms as ps
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
import time
from pymoo.util.misc import find_duplicates
from utils.fps import FPS
from config import *
import utils.survival as survival
from visualization import output_svm

# log.basicConfig(handlers=[ log.StreamHandler()], level=log.INFO)

class NSGAII_SVM_SIM(object):

    algorithm_name = "NSGA-II-SVM"

    def __init__(self,
                 problem: problem,
                 config: SearchConfiguration):

        self.problem = problem
        self.config = config
        self.res = None

        log.info(f"Initialized algorithm with config: {config.__dict__}")
        
    def run(self) -> ResultExtended:
        """
        Runs the NSGA-II-SVM algorithm for sampling optimization.

        This method implements the NSGA-II-SVM algorithm for generating optimized samples.
        It performs multi-objective optimization using the NSGA-II algorithm and generates
        additional samples using an SVM classifier in critical regions of the search space.
        
        Config:
        -------
        The algorithm configuration should contain the following parameters:

            - `config.population_size` (int):   Size of the initial population. Not counted towards
                                            the total number of evaluations.
            - `config.n_generations` (int):     Number of (inner) generations the NSGA-II algorithm runs.
            - `config.new_samples` (int):       Number of samples generated with the SVM model in
                                            critical regions.
            - `config.n_func_evals_lim` (int):  Limits the execution of the algorithm. The algorithm
                                            stops after at least `config.n_func_evals_lim` samples
                                            have been evaluated.
            - `config.num_offsprings` (int):    Used for initialization of pymoo NSGA-II algorithm.
            - `config.maximal_execution_time` (str or None): The maximum execution time for the algorithm in the format "HH:MM:SS".
            
        Either `config.maximal_execution_time` or `config.n_func_evals_lim` can be given, but not both.

        Returns:
        -------
        ResultExtended:
            An extended pymoo result object containing the best population, all evaluated samples, and other
            relevant information.
        
        Notes:
        ------
        This method implements the NSGA-II-SVM algorithm, which combines the NSGA-II algorithm
        with an SVM-based sampling strategy to generate new samples. It iterates through
        outer iterations, where NSGA-II generates the best population after `config.n_generations`
        inner iterations. NSGA-II always uses the current `config.population_size` best samples in the total population,
        as an initial sampling. Then new samples over all `config.n_generations` from NSGA-II are merged with
        the current total population (`all_population`), to train the SVM classifier. The SVM classifier predicts critical
        regions in the search space, and new samples are generated in these regions. Only the newly generated
        samples for each outer iteration are stored in the history list of the final result object.
        
        The algorithm keeps track of the total number of function evaluations (`n_func_evals`)
        and stops when the limit specified by `config.n_func_evals_lim` is reached.
        
        The final population for the result is determined by selecting the best samples from
        `all_population` over all outer iterations.
        """
        # Start the timer
        start_time = time.time()
        
        problem = self.problem
        config = self.config
        max_time = None

        if(config.maximal_execution_time is not None):
            t = config.maximal_execution_time
            # Convert the time string to a datetime object
            time_obj = time.strptime(t, '%H:%M:%S')
            
            # Calculate the total seconds from the time object
            max_time = time_obj.tm_hour * 3600 + time_obj.tm_min * 60 + time_obj.tm_sec
        
        # Create a Result object for output
        result = ResultExtended()
        
        '''Temporary hard coded parameters''' 
        population_size = config.population_size
        max_iterations = config.n_generations
        new_samples_svm = config.new_samples*population_size if isinstance(config.new_samples,float) else config.new_samples 

        all_population = PopulationExtended()
        best_population = PopulationExtended()
        
        self.termination = get_termination("n_gen", max_iterations)

        #  create initial population
        # sampling = LHS()  # Latin Hypercube Sampling
        sampling = FPS()
        initial_population = sampling(problem, population_size)
        evaluate_individuals(initial_population, problem)
        
        population = initial_population
        iteration = 0
        n_func_evals = 0
        hist = list()
        
        elapsed_time = time.time() - start_time
        termination_cond = check_termination_cond(config.n_func_evals_lim, n_func_evals, max_time, elapsed_time)
        
        sampled_pop_store = []
        svm_model_store = []

        while termination_cond:
            log.info(f"Running iteration {iteration}")
            # log.info(f"size of seed population: {len(population)}")
            res = run_NSGA2(problem,
                            population=population,
                            population_size=population_size,
                            termination = self.termination,
                            num_offsprings=config.num_offsprings,
                            )
            # Get new population from NSGA2 algorithm. This includes all individuals over all inner generations
            new_pop_nsga2 = update_all_population(PopulationExtended(), res.history)
            # Update the total population
            all_population = update_all_population(all_population, res.history)
            
            grid_search_cv = None

            #### Create svm model of search space; take all individuals not to lose any information
            samples_pop = sample_svm_random(problem = problem, 
                                                    population = all_population, 
                                                    svm_model_store = svm_model_store,
                                                    n_samples= new_samples_svm)
            # evalute new samples
            evaluated_sampled = evaluate_individuals(samples_pop, problem)

            # store for result object
            sampled_pop_store.append(evaluated_sampled)
            
            # Get all new individuals for this outer iteration
            new_population = Population.merge(evaluated_sampled, new_pop_nsga2)
            temp_best_population = get_nondominated_population(new_population)
            
            # Update all population with new samples from SVM model
            all_population = Population.merge(all_population, evaluated_sampled)
            
            # n number of Nds population is used for each iteration to find new samples from NSGA-II
            # population = get_individuals_rankwise(all_population, population_size)

            # group first by criticality and order by crowding distance
            population = survival.RankAndLabelAndCrowdingSurvival().do(
                                                            problem=res.problem, 
                                                            pop=all_population, 
                                                            n_survive = population_size)


            best_population = get_nondominated_population(population)
            
            
            iteration += 1
            n_func_evals += len(new_population)

            #### statistics
            # crit, _ = new_population.divide_critical_non_critical()
            # log.info(f"Ratio critical sampled with svm: {len(crit)/len(new_population)}")
            
            crit, _ = evaluated_sampled.divide_critical_non_critical()
            log.info(f"Ratio critical sampled using SVM: {len(crit)/len(evaluated_sampled)}")

            # Create evaluator and algorithm object for res.history
            hist_temp_algorithm = Algorithm()
            hist_temp_eval = Evaluator()
            # Total number of func evals over all iterations
            hist_temp_eval.n_eval=n_func_evals
            
            hist_temp_algorithm.problem = problem
            hist_temp_algorithm.evaluator = hist_temp_eval
            # New samples generated for this iteration
            hist_temp_algorithm.pop = new_population
            hist_temp_algorithm.opt = temp_best_population
            hist_temp_algorithm.n_iter = iteration
            
            hist.append(hist_temp_algorithm)
            
            # Update termination criterion before next iteration begins
            elapsed_time = time.time() - start_time
            termination_cond = check_termination_cond(config.n_func_evals_lim, n_func_evals, max_time, elapsed_time)
        
        ##############
        # Stop the timer
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        ##############

        result.problem = problem
        result.algorithm = Algorithm()
        # Best population after aglorithm terminates of size population_size
        result.pop = population
        result.X = population.get("X")
        result.F = population.get("F")
        result.CV = population.get("CV")
        result.G = population.get("G")
        result.H = population.get("H")
        result.opt = best_population
        result.start_time = start_time
        result.end_time = end_time
        result.exec_time = elapsed_time
        result.history = hist
        
        result.additional_data["svm_model"] = svm_model_store
        result.additional_data["sampled_pop"] = sampled_pop_store

        self.res = result
        return result

    def write_results(self, 
                        ref_point_hv, 
                        ideal, 
                        nadir, 
                        results_folder = RESULTS_FOLDER):
        
        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info("Result object is None. Execute algorithm first, before writing results.")
            return
        log.info(f"=====[{self.algorithm_name}] Writing results...")
        config = self.config
        res = self.res
        algorithm_parameters = {
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "New samples": str(config.new_samples),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation)
        }
        
        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name, is_experimental=EXPERIMENTAL_MODE)
        
        output.igd_analysis(res, 
          save_folder,
          critical_only=True)
        output.gd_analysis(res,
          save_folder,
          critical_only=True)
        output.hypervolume_analysis(res, 
          save_folder, 
          critical_only=True,
          ref_point_hv=ref_point_hv, 
          ideal=ideal, 
          nadir=nadir)  
        
        output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)
        output.all_critical_individuals(res, save_folder)
        output.write_generations(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

        output_svm.write_svm_data(res, save_folder)

        #persist results object
        res.persist(save_folder + "backup")

def check_termination_cond(max_n_func_evals = None, n_func_evals = 0, max_time = None, elapsed_time = None):
    """
    Check the termination condition for an algorithm.

    This method is used to determine if an algorithm should terminate based on either the maximum number of function evaluations
    or the maximum execution time.

    Parameters:
        max_n_func_evals (int or None): The maximum number of function evaluations.
        n_func_evals (int): The current number of function evaluations.
        max_time (float or None): The maximum execution time in seconds.
        elapsed_time (float): The elapsed time in seconds since the start of the algorithm.

    Returns:
        bool: True if the termination condition is met, False otherwise.

    Notes:
        - If `max_n_func_evals` is not None and `max_time` is None, the termination condition is based on the number of generations.
        - If `max_time` is not None and `max_n_func_evals` is None, the termination condition is based on the elapsed time.
        - If both `max_n_func_evals` and `max_time` are given, an error is logged and None is returned.
    """
    if(max_n_func_evals is not None and max_time is None):
        return n_func_evals < max_n_func_evals
    elif(max_n_func_evals is None and max_time is not None):
        return elapsed_time < max_time
    else:
        log.error("Error: Max_n_func_evals and maximum execution time were given, as termination criterion. Only one is supported.")
        return None

def sample_svm_random(problem, population, svm_model_store, n_samples = 10):
    """
    Generates new critical samples using an SVM classifier and random sampling.

    This method uses a trained SVM classifier to generate new samples that are predicted
    as critical (class label = 1). If there are not enough samples of both critical
    or non-critical classes, random sampling is done instead.

    Parameters:
    ----------
    problem : pymoo.problem.Problem
        The optimization problem for which samples are generated.
    population : pymoo.model.Population
        The existing population of solutions.
    n_samples : int, optional
        The number of new samples to generate, by default 10.

    Returns:
    -------
    pymoo.model.Population
        A new population containing the generated predicted critical samples.

    Notes:
    -----
    This method generates new samples by first training an SVM classifier on the existing
    population. The SVM classifier is trained using grid search with cross-validation to
    find the best hyperparameters. Then, random sampling is performed, and samples
    predicted as critical by the SVM are added to the result population.

    If the number of critical or non-critical samples is less than or equal to the
    dimensionality of the problem, random sampling is performed instead to ensure
    diversity in the generated samples.
    """
    log.info(f"Sampling in svm region for {n_samples} samples...")
    X, y = population.get("X","CB")
    dim = X.shape[1]
    crit, ncrit = population.divide_critical_non_critical()
    
    if(len(crit) <= dim or len(ncrit) <= dim):
        # Not enough individuals per class
        log.warning("Warning: Not enough samples per class! Random sampling done instead.")
        return FloatRandomSampling().do(problem,n_samples)
    
    # Define the parameter grid for grid search
    param_grid = {'C': [1, 10, 100, 1000],
                'gamma': [0.001, 0.1, 1, 10]}

    # Perform grid search with cross-validation
    svm_classifier = GridSearchCV(
                            SVC(kernel='rbf'), 
                            param_grid, 
                            cv=5
                    )
    # Train model
    svm_classifier.fit(X,y)
    
    svm_model_store.append(svm_classifier)

    samples = Population()
    i = 0
    while i < n_samples:
        # Generate random sample and predict criticality label
        sample = FloatRandomSampling().do(problem,1)[0]
        if svm_classifier.predict([sample.get("X")])[0] == 1:
            i += 1
            samples = Population.merge(samples, sample)
    return samples #svm_classifier.best_estimator_   

def extract_bounds(input_bounds):
    lower_bounds = [bound[0] for bound in input_bounds]
    upper_bounds = [bound[1] for bound in input_bounds]
    return np.array(lower_bounds), np.array(upper_bounds)


def find_smallest_difference_tuple(data):
    smallest_difference = float('inf')
    smallest_tuple = None

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                first, second = data[i]
                third, fourth = data[j]
                difference = abs(first - fourth)

                if difference < smallest_difference:
                    smallest_difference = difference
                    smallest_tuple =np.array( [first, fourth])

    return smallest_tuple

def update_all_population(all_population, history):   
    for generation in history:
        all_population = PopulationExtended.merge(
                    all_population, generation.pop)
    return all_population

def update_pop_after_sampling_nds(old_pop, sampled_pop, n_replace):
    pop_all = Population.merge(old_pop, sampled_pop)
    pop_new = get_individuals_rankwise(pop_all,n_replace)
    return pop_new

def update_pop_after_sampling(old_pop, sampled_pop, n_replace):
     # replace the "n_replace" individuals by individuals sampled from the regions
    # if not enough individuals in nds sets, sample
    # if len(pop_add) < n_replace:
    #     rand_sampled_pop = FloatRandomSampling().do(sub_problem, n_replace - len(pop_add))
    #     pop_add = PopulationExtended.merge(pop_add, rand_sampled_pop)
    pop_new = copy.deepcopy(old_pop)
    for i in range(0,n_replace):
        pop_new[-(i+1)] = sampled_pop[i]
        
    return pop_new

def filter_duplicates_pop(pop):
    # Filters duplicates from population. (Method is currently not used)
    X = pop.get("X")
    is_unique = np.where(np.logical_not(find_duplicates(X, epsilon=1e-32)))[0]
    return Population.new("X", X[is_unique])

def run_NSGA2(problem, population_size,
                population, 
                termination,
                num_offsprings=None, 
                prob_crossover=0.7, 
                eta_crossover=20, 
                prob_mutation=0.5, 
                eta_mutation=15) -> ResultExtended:
    """
     Run the NSGA-II algorithm for multi-objective optimization.

    This method applies the NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm for multi-objective
    optimization. It takes a problem instance, an initial population, and various algorithmic parameters to
    execute the optimization process.

    Args:
    -----
    problem (pymoo.model.problem.Problem): The optimization problem instance to be solved.
    population_size (int): The size of the population.
    population (pymoo.factory): The initial population for the algorithm.
    termination (pymoo.factory): The termination criterion for stopping the algorithm.
    num_offsprings (int, optional): The number of offspring individuals to be generated in each generation.
                                    Default is None (use population_size).
    prob_crossover (float, optional): The probability of crossover (SBX) operation during mating.
                                      Default is 0.7.
    eta_crossover (float, optional): The parameter controlling the strength of crossover.
                                     Default is 20.
    prob_mutation (float, optional): The probability of mutation (Polynomial Mutation) operation during mating.
                                     Default is 0.5.
    eta_mutation (float, optional): The parameter controlling the strength of mutation.
                                    Default is 15.

    Returns:
    --------
    res (ResultExtended): The result object containing the information about the optimization process.
    """
    #### apply genetic algorithm
    algorithm = NSGA2(
            pop_size=population_size,
            n_offsprings=num_offsprings,
            sampling=population,
            crossover=SBX(prob=prob_crossover, eta=eta_crossover),
            mutation=PM(prob=prob_mutation, eta=eta_mutation),
            eliminate_duplicates=True)

    res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=False)
    return res