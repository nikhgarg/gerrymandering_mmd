This repository contains code for the paper following paper:

@inproceedings{garg2022combatting,
  title={Combatting gerrymandering with social choice: The design of multi-member districts},
  author={Garg, Nikhil and Gurnee, Wes and Rothschild, David and Shmoys, David},
  booktitle={Proceedings of the 23rd ACM Conference on Economics and Computation},
  pages={560--561},
  year={2022}
}

Full version on arXiv: https://arxiv.org/abs/2107.07083

This repository is organized as follows:

1. `optimize/` contains the optimization code that generates optimal multi-member maps for each state, for each objective function. This algorithm is based on the algorithm described in the following paper:

    @inproceedings{gurnee2021fairmandering,
    title={Fairmandering: A column generation heuristic for fairness-optimized political districting},
    author={Gurnee, Wes and Shmoys, David B},
    booktitle={SIAM Conference on Applied and Computational Discrete Algorithms (ACDA21)},
    pages={88--99},
    year={2021},
    organization={SIAM}
    }

    We have uploaded the output maps used in the paper in the following Box folder: https://cornell.box.com/s/xrzcblq4wg7bpe51q26cgpklwjuwz3ph

2. The remainder of the code uses the generated maps to compute the results in the paper, including:
     - visualization of various maps
     - computing the proportionality and competitiveness results for each state and overall, for each set of maps (fair, random, and optimized for each party)
     - setting up and running STV elections for the intra-party results. 

Figure reproduction note:
 - `usevisualization_202601_withgerrymanderinggap.ipynb` is the main notebook for the proportionality, gerrymandering, competitiveness, and intra-party plots built from cached organized outputs.
 - `20260103_plot_proportionality_by_noise_rightmaps.ipynb` reproduces the voter-noise / crossover robustness plots.
 - `20260418_regenerate_hr4000_figs_4_9_10.ipynb` is a notebook for the HR4000 / Fair Representation Act comparison plots from the currently available cached artifacts.
 - `visualize_plans.ipynb` produces the Texas and Illinois appendix plan panels.
 - Historical pipeline launch scripts used to generate cached outputs are archived in `server_run_files/`.


data/ description:
 - data/voterfile_withscores_noised.csv: a noisy, sampled version of the individual level voter file data used for the intra-party analysis in the paper
