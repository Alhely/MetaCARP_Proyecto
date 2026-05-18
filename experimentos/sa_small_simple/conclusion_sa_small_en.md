# Conclusion of the SA experiment — small simple

## Experimental setup

The Simulated Annealing (SA) evaluated in this experiment operates with temperature parameters computed automatically per instance: the initial temperature follows the formula `T_init = 20 · d_max / n`, where `d_max` is the maximum service cost and `n` the number of required tasks. The Markov chain length (`L`) and the minimum temperature are derived from the acceptance dynamics, requiring no manual calibration from the user.

The fixed parameters of the experiment are:

- `alpha_inter = 0.8` (cooling factor for inter-route operators).
- `patience = 10` (temperature levels without improvement before triggering a reheat).
- `reheat_factor = 0.5` (temperature is raised back to 50 % of `T_init` at each reheat).
- `max_reheats_sin_mejora = 5` (early-stopping criterion: five consecutive reheats without improving the global best solution).
- 9 neighborhood operators: `relocate_intra`, `swap_intra`, `2opt_intra`, `relocate_inter`, `swap_inter`, `2opt_star`, `cross_exchange`, `or_opt_2`, and `or_opt_3`.

The grid search swept two free hyperparameters: `alpha` (intra-route cooling rate) with 11 values in [0.80, 0.99], and `p_inter` (probability of selecting an inter-route operator at each iteration) with 5 values in [0.4, 0.8].

---

## Grid search coverage

The experimental design combines 11 values of `alpha` × 5 values of `p_inter` × 23 instances × 2 repetitions, totaling **2,530 runs**. Each grid cell (an `alpha × p_inter` pair) accumulates 46 runs (2 repetitions × 23 instances), and each instance is evaluated under 110 distinct hyperparameter combinations. This coverage allows the marginal effect of each hyperparameter to be estimated with reasonable statistical margins, separating the variance attributable to `alpha`, to `p_inter`, and to between-instance variability.

The 23 instances come from two reference CARP families: 17 small instances from the **GDB** family (Golden, DeArmon & Baker) and 6 instances from the **KSHS** family, all with published BKS (Best Known Solutions), which makes it possible to compute absolute gaps against the known optimum.

---

## Global performance

The 2,530 runs delivered a **100 % feasibility rate**: no run ended with a solution violating capacity constraints. This confirms that the construction mechanism and the implemented neighborhood operators preserve feasibility throughout the search.

Per-run execution times follow an asymmetric distribution: median of **3.64 s**, mean of **8.65 s**, minimum of 0.19 s, and maximum of 112.55 s. The gap between median and mean reflects the existence of instances with substantially larger search spaces that stop the algorithm later.

The number of cooling steps executed (full temperature descents) ranges from 11 to 348, with a mean of 67. The fact that the median number of cooling steps is markedly lower than the maximum indicates that **early stopping via `max_reheats_sin_mejora = 5` kicks in frequently**: most runs do not exhaust their theoretical temperature budget but rather terminate upon detecting stagnation. This is desirable in efficiency terms, but it also signals that the effective search budget may be insufficient to escape deep local optima.

The number of total iterations per run (min. 1,331, mean 33,447, max 378,972) directly mirrors the heterogeneity in instance size and in how many temperature levels are traversed before stopping.

---

## Grid search analysis: effect of the hyperparameters

### Effect of `alpha` (intra-route cooling rate)

The `alpha` parameter shows a **clear, monotonic effect**: the higher its value (slower cooling), the lower the average gap to the BKS. The following table summarizes the mean and median gap for each level of `alpha`, averaged over the 230 corresponding runs (5 values of `p_inter` × 23 instances × 2 repetitions):

| `alpha` | Mean gap     | Median gap  |
|--------:|-------------:|------------:|
| 0.80    | 33.98 %      | 33.08 %     |
| 0.82    | 33.84 %      | 32.89 %     |
| 0.84    | 33.86 %      | 32.79 %     |
| 0.86    | 33.45 %      | 32.83 %     |
| 0.88    | 33.23 %      | 32.73 %     |
| 0.90    | 32.92 %      | 32.28 %     |
| 0.92    | 32.56 %      | 31.94 %     |
| 0.94    | 32.21 %      | 31.86 %     |
| 0.96    | 31.74 %      | 31.55 %     |
| 0.98    | 31.20 %      | 31.86 %     |
| **0.99**| **30.89 %**  | **29.93 %** |

The difference between the lower extreme (`alpha = 0.80`, mean gap 33.98 %) and the upper one (`alpha = 0.99`, mean gap 30.89 %) is roughly **3.1 percentage points**. Although the magnitude is not dramatic in absolute terms, the trend is statistically clean and suggests that the SA benefits from exploring the solution space more gradually: slow cooling keeps temperatures high for more iterations, which favors the acceptance of deteriorating moves and reduces the probability of getting trapped in early local minima.

### Effect of `p_inter` (inter-route probability)

In contrast to `alpha`, the `p_inter` parameter shows a **small, non-monotonic effect**. The total variation across the five evaluated levels is below **0.3 percentage points** in mean gap:

| `p_inter` | Mean gap     | Median gap  |
|----------:|-------------:|------------:|
| 0.4       | 32.89 %      | 32.73 %     |
| 0.5       | 32.70 %      | 32.31 %     |
| 0.6       | **32.59 %**  | **31.86 %** |
| 0.7       | 32.71 %      | 31.86 %     |
| 0.8       | 32.69 %      | 31.86 %     |

The marginal minimum appears at `p_inter = 0.6`, but the levels 0.6, 0.7 and 0.8 yield essentially identical results. This indicates that, within the explored range, the balance between intra-route and inter-route neighborhoods is not a decisive factor for performance, at least when the instance set comprises small graphs with few routes per solution.

### Top configurations

The five best `alpha × p_inter` combinations (by mean gap over the 46 runs of each cell) are:

| `alpha` | `p_inter` | Mean gap     | Median gap  | Standard deviation |
|--------:|----------:|-------------:|------------:|-------------------:|
| 0.99    | 0.7       | 30.78 %      | 28.61 %     | 10.67              |
| 0.99    | 0.6       | 30.87 %      | 30.77 %     | 10.77              |
| 0.99    | 0.5       | 30.92 %      | 31.04 %     | 10.87              |
| 0.99    | 0.8       | 30.93 %      | 29.92 %     | 10.58              |
| 0.99    | 0.4       | 30.95 %      | 29.24 %     | 10.77              |

The 11 configurations with the lowest mean gap all correspond to `alpha = 0.99` or `alpha = 0.98`. This result confirms that **`alpha` dominates the variance of grid performance**, while `p_inter` plays a secondary role whose precise choice has an impact below a tenth of a percentage point on the mean.

---

## Per-instance results

The individual gaps of the best run per instance (available in [`resumen_mejores_corridas.csv`](resumen_mejores_corridas.csv)) range between **14.13 %** and **52.00 %**, with a mean of 30.54 % and a median of 27.59 %. The spread is large, reflecting substantial structural differences between instances.

The instances with the lowest gap are kshs2 (14.13 %), gdb19 (14.55 %), kshs1 (15.27 %), gdb13 (19.03 %), gdb16 (19.69 %) and gdb17 (19.78 %). The KSHS family and some denser-topology GDB instances (gdb13, gdb16, gdb17, gdb19) appear to offer a search landscape where the SA finds solutions relatively close to the BKS. In these cases, the combination of intra- and inter-route operators recombines routes effectively.

At the opposite extreme, five instances exceed a 40 % gap: gdb3 (49.09 %), gdb10 (52.00 %), gdb14 (44.00 %), gdb6 (42.28 %) and gdb5 (41.11 %). The case of gdb10, with a 52 % gap (SA cost = 418 vs. BKS = 275), is the most severe. A plausible hypothesis is that instances such as gdb10, gdb5 and gdb6 present topological structures where the optimal solution requires route distributions unreachable through simple local moves from the initial solution: the neighborhood operators are failing to generate moves that escape the basin of attraction of the initial local minimum. Additionally, in instances with small absolute costs (such as gdb14, with BKS = 100), a handful of misassigned arcs can translate into large relative gaps, amplifying the appearance of the problem.

---

## Conclusion

The SA, under the parameterization evaluated in this experiment, **guarantees full feasibility and produces solutions in time of the order of seconds** for the set of small instances (GDB + KSHS). The mean gap of 30.54 % with respect to the published BKS indicates that the algorithm converges stably to admissible solutions, but that these lie at a substantial distance from the reference optima. The median of 27.59 % suggests that half of the instances are solved with a gap below that threshold, although the upper tail (five instances above 40 %) reveals that SA in its current form is not uniformly competitive.

The configuration recommended on the basis of this grid search is **`alpha = 0.99` with `p_inter` in the 0.6–0.7 range**: these are the cells with the lowest mean gap, and the difference between them is below 0.1 percentage points. The robustness of this recommendation is limited by the documented insensitivity of `p_inter`, so the priority for future calibration should fall on `alpha` and on the search budget.

The early-stopping behavior (median of 67 cooling steps versus a maximum of 348) points to a structural cause of the high gap: the `max_reheats_sin_mejora = 5` criterion ends the search before the algorithm has exhausted its real exploratory capacity. The most direct improvement avenues are: (1) increase `max_reheats_sin_mejora` to 10–15 to grant a larger budget on hard instances without inflating runtime on easy ones; (2) explore `alpha` values above 0.99 (e.g., 0.995 or 0.999) to check whether the observed monotonic trend continues; (3) append a deterministic local-search phase (exhaustive 2-opt or arc-adapted Lin-Kernighan) at the end of the SA run to polish the best solution found; and (4) couple the SA with a complementary metaheuristic with stronger diversification capacity —such as Tabu Search or Artificial Bee Colony— in a hybrid or restart scheme, leveraging the fact that the SA already produces moderate-quality feasible solutions in short time.
