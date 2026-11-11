# Statistical Analysis Summary

- Significance threshold: alpha = 0.05
- Bootstrap resamples (grouped & stratified): 2000
- Multiple comparison adjustment: Holm-Bonferroni within each track/test family
- Caveat (Seeds vs Cohorts): Multi-seed tests (Paired t-test, Wilcoxon, Friedman) capture optimization variance on fixed dataset splits, NOT independent patient cohorts.
- Caveat (Wilcoxon Power): At n=5 seeds, two-sided Wilcoxon signed-rank has minimum achievable p = 0.0625 (> 0.05 before Holm correction).
- Caveat (McNemar Clustering): Sample-level McNemar assumes independent test errors; clustered OCT slices (from few inferred sequences) violate i.i.d. assumptions and are flagged in pairwise notes.

- Total hypothesis tests conducted: 149 (significant at Holm-adjusted alpha=0.05: 87)