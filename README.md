# About

This repository holds the implementation and evaluation results of the NSGA-II-SVM approach presented in the paper with the title submitted at DeepTest@ICSE24: "Guiding the Search Towards Failure-Inducing Test Inputs Using Support Vector Machines".

## Algorithm

The reflection study has been conducted with the support of the open-source and modular testing framework [OpenSBT](https://git.fortiss.org/opensbt/opensbt-core). The implementation is provided [here](algorithm/kernel_svm.py). The algorithm integrated into OpenSBT can be found [here](application/opensbt-svm.zip).

## Results Combined

The *Hypervolume/General Distance/Spread* analysis results after 10 runs are available here: 

<img src="results/subplots_combined_gd.png" alt="results_10_runs" width="600"/>

The detailed *Hypervolume* analysis results are available here: [HV](/results/hv/)

The *Spread* analysis results are available here: [Spread](/results/sp/) 

The *General Distance* analysis results are available here: [GD](/results/gd/) 
The estimated Pareto front is available here: [PF](/results/estimated_pf.csv)

The results related to the number of distinct critical testcases are available here: [Distinct Failures](/results/n_crit/) 

The corresponding statistical analysis results for the metric `METRICNAME` can be found in the file `METRICNAME_significance.csv` in the corresponding metric folder.

Note, that in the graphs more evaluations are visible done by NSGA-II-DT then used for the comparison (1000), as the execution time of NSGA-II-DT is not controllable because of the flexible population size.

## Results Ratio Sampling

Below are provided results which show the ratio of critical test cases from all sampled tests in an SVM region over time (9 SVM generations per run) for all runs and on average. This metric should assess the guidance of the SVM Model with increasing number of SVM iterations. 

<center>
<img src="results/ratio/svm_ratio.png" alt="Ratio_SVM" width="600"/>
</center>

The detailed results are provided here: [Results ratio](results/ratio/svm_ratio.txt)

## Additional Material
### Motivation
<center>
    <p float="left">
        <img src="results/motivation/DT_example.png" alt="Exemplary Decision Tree algorithm" width="300">
        <img src="results/motivation/SVM_example.png" alt="Exemplary SVM algorithm" width="300">
    </p>
     <figcaption>
     Illustration of difference between boundary identification by a Decision Tree and Support Vector Machine for data from two classes(circle/triangle positive/negative data).
     </figcaption>
</center>

### Approach Overview

<div style="margin: 80;"></div>
<center>
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="results/algorithm/kernel_svm_1_c-1.png" alt="First step of NSGA-II-SVM" width="125" style="margin: 1;">
        <img src="results/algorithm/kernel_svm_2_c-1.png" alt="Second step of NSGA-II-SVM" width="125" style="margin: 1;">
        <img src="results/algorithm/kernel_svm_3_c-1.png" alt="New iteration with first step of NSGA-II-SVM" width="125" style="margin: 1;">
        <img src="results/algorithm/kernel_svm_4_c-1.png" alt="New iteration with second step of NSGA-II-SVM" width="125" style="margin: 1;">
        <img src="results/algorithm/kernel_svm_legend_c-1.png" alt="Legend for NSGA-II-SVM algorithm steps" width="125" style="margin: 1;">
    </div>
   <div style="margin: 60;"></div>
    <figcaption>
     Overview of the main steps of NSGA-II-SVM. 1. Figure: Genetic search using NSGA-II. 2. Figure: Learned SVM-model for failing region prediction and sampling inside region. 3. Figure: Genetic search with NSGA-II in subsequent iteration using evaluated samples and best solutions found. 4. Figure: Refined SVM-model learning. Arrows indicate genetic operations.
     </figcaption>
</center>
