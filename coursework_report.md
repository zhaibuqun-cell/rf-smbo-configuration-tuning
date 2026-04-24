# Configuration Performance Tuning with a Random-Forest Sequential Model

## Introduction

This coursework presents a budget-limited intelligent software engineering tool for Lab 3: Configuration Performance Tuning. The task is to search for high-performing configurations while consuming only a small number of measurements, which mirrors the practical setting where each configuration evaluation may be expensive.

The lab baseline is **Random Search**, while the proposed tool is **Random-Forest Sequential Model-Based Optimisation (RF-SMBO)**. Under the final budget of 50 measurements, RF-SMBO reduces the overall mean normalised gap from 0.0487 to 0.0113, a reduction of 76.87%. However, this result should not be stated too strongly: RF-SMBO overall improves the mean normalised gap mainly because it performs much better on **LLVM** and **x264**, but it underperforms **Random Search** on **7z**.

## Related Work

Random Search is a natural baseline for black-box optimisation because it is simple, budget-aware, and surprisingly competitive in high-dimensional spaces when compared with naive exhaustive strategies [1]. For this reason it is appropriate as the baseline in Lab 3, where the search budget is explicitly limited and every measurement is treated as costly.

Sequential Model-Based Optimisation (SMBO) improves on blind sampling by learning a surrogate model from past observations and using that surrogate to guide the next measurement [2]. Bayesian optimisation is a closely related family of methods that uses predictive uncertainty to balance exploration and exploitation [3]. This idea is directly relevant to the current coursework because the objective is not to evaluate every configuration, but to spend a small budget on measurements that are likely to be informative.

Random-forest surrogates are especially appealing in configurable software systems because they can handle non-linear interactions, mixed discrete or integer-like options, and irregular performance landscapes. Prior work on software configuration tuning shows that uncertainty-aware and surrogate-guided optimisation can be effective for performance-sensitive systems [4], and that even imperfect learners can still help identify useful configurations in search-based software engineering settings [5]. The present project builds on that idea by using a random forest as the surrogate inside an SMBO loop.

## Solution

The search space is defined by the valid rows already present in each dataset. Each row represents one legal candidate configuration. Runtime values are **not** used in advance to rank all candidates; instead, a runtime value is only revealed when that row is selected for measurement. After each new measurement, RF-SMBO retrains a `RandomForestRegressor` on the observed configurations and their measured runtimes. A random forest is suitable here because it combines many decision-tree regressors and averages their predictions, which improves prediction stability and helps control over-fitting while still modelling non-linear option interactions.

### Algorithm 1: RF-SMBO for Configuration Performance Tuning

```text
1. Initialise the observed set with six random configurations.
2. Measure those configurations and store the revealed runtimes.
3. Train RandomForestRegressor on the observed configurations.
4. Predict mean and uncertainty for unmeasured configurations.
5. Compute acquisition score = predicted_mean - 0.5 * predicted_std.
6. With 10% probability, select the most uncertain candidate.
7. Otherwise, select the candidate with the lowest acquisition score.
8. Measure the selected configuration and update the observed set.
9. Repeat until the measurement budget is exhausted.
10. Return the best observed configuration.
```

The baseline Random Search samples unseen valid configurations uniformly at random without replacement.

## Setup

The final evaluation uses three Lab 3 systems: **7z** (68,640 configurations, 8 options), **LLVM** (65,536 configurations, 16 options), and **x264** (4,608 configurations, 10 options). All objectives are minimisation problems. Budgets are 10, 20, 30, 40, and 50 measurements, and each algorithm is repeated 30 times with paired seeds for fair comparison.

The primary metric is the **normalised gap to the optimum**, defined as `(best_found - optimum) / (worst - optimum)`, where lower is better. The secondary metric is the best runtime found under the given budget. Statistical significance is assessed using a **one-sided paired Wilcoxon signed-rank test**, testing whether RF-SMBO achieves a lower normalised gap than Random Search.

The global optimum and worst performance values are used **only for post-hoc evaluation and normalisation**. They are **never** used by Random Search or RF-SMBO during the search process.

## Experiments

Figure 1 shows convergence as the budget increases. The proposed method generally improves faster than the baseline because the surrogate becomes more informative after each new measurement.

Table 1 summarises the final-budget results at 50 measurements, while Figure 2 shows the per-system differences. RF-SMBO improves clearly on **LLVM** and **x264** but loses on **7z**, so the overall result must be interpreted as a mixed but promising outcome rather than a universal win.

| System | Random best | RF-SMBO best | Random gap | RF-SMBO gap | Gap reduction (%) | p-value | win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7z | 5405.76 | 6898.31 | 0.0029 | 0.0064 | -123.42 | 0.9241 | 46.7% |
| LLVM | 59231.21 | 53682.79 | 0.1351 | 0.0272 | 79.88 | 9.31e-10 | 100.0% |
| x264 | 22.9545 | 21.5853 | 0.0080 | 0.0002 | 97.90 | 9.10e-07 | 96.7% |

On **LLVM**, RF-SMBO reduces the mean best runtime from 59231.21 to 53682.79 and reduces the normalised gap by 79.88% (`p = 9.31e-10`). On **x264**, the runtime improves from 22.9545 to 21.5853, with a 97.90% reduction in normalised gap (`p = 9.10e-07`). In contrast, **7z** remains a failure case where RF-SMBO is worse than the baseline.

## Reflection and Conclusion

The negative result on **7z** should be kept rather than hidden, because it is a scientifically valid result. It shows that the surrogate model is not always reliable, even when the same method works well on other systems. Several factors may explain this outcome. First, the 7z search landscape may be rugged or deceptive, making it harder for a small surrogate model to rank candidates well. Second, the measurement budget may provide too little training data for accurate model fitting. Third, the uncertainty estimate derived from tree disagreement in the random forest is only an approximation and may not reflect the true search uncertainty. Fourth, some configuration spaces may reward broader exploration more than model-guided exploitation.

Despite that limitation, RF-SMBO is still useful overall because it delivers strong improvements on LLVM and x264 under the same limited budgets. Future work should investigate **Expected Improvement** or rank-based acquisition, an adaptive exploration coefficient instead of the fixed value 0.5, comparisons against stronger baselines such as local search, genetic algorithms, or A-T-EGLS, multi-objective runtime-energy tuning, and additional systems from Lab 3.

In conclusion, RF-SMBO improves the overall mean normalised gap under a limited measurement budget, but the benefit is driven by strong gains on LLVM and x264 rather than universal superiority. The 7z failure case therefore strengthens the reflection section and makes the submission more scientifically credible.

## Artifact

Artifact link: https://github.com/zhaibuqun-cell/rf-smbo-configuration-tuning

The repository includes source code, raw CSV outputs, generated figures, `requirements.pdf`, `manual.pdf`, and `replication.pdf`.

## References

[1] J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," *Journal of Machine Learning Research*, vol. 13, no. 10, pp. 281-305, 2012. Available: https://www.jmlr.org/papers/v13/bergstra12a.html

[2] F. Hutter, H. H. Hoos, and K. Leyton-Brown, "Sequential Model-Based Optimization for General Algorithm Configuration," *LION 5 / UBC Technical Report TR-2010-10*, 2011. Available: https://www.cs.ubc.ca/tr/2010/tr-2010-10

[3] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," in *Advances in Neural Information Processing Systems*, 2012. Available: https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms

[4] P. Jamshidi and G. Casale, "An Uncertainty-Aware Approach to Optimal Configuration of Stream Processing Systems," in *MASCOTS*, 2016. Available: https://pooyanjamshidi.github.io/resources/papers/bo4co.pdf

[5] V. Nair, T. Menzies, N. Siegmund, and S. Apel, "Using Bad Learners to find Good Configurations," in *ESEC/FSE*, 2017. Available: https://arxiv.org/abs/1702.05701

[6] ideas-labo, "ISE Lab 3 repository." Available: https://github.com/ideas-labo/ISE/tree/main/lab3
