# Caco-2 apparent-permeability modeling

## Dataset curation

Three publicly available Caco-2 apparent-permeability collections were
curated from Wang et al. (2016), Wang and Chen (2020), and Wang et al. (2020)
and are referred to collectively as **Public3**. Structures were standardized
as canonical isomeric SMILES and assigned full standard InChIKeys. Permeability
measurements were converted to a common endpoint, log10 apical-to-basolateral
apparent permeability in cm/s (`Log10_Caco_Papp_AB_cm_s`). In particular,
values reported by Wang et al. (2020) as log10(Papp x 10^6) were shifted by
-6 to obtain log10(Papp in cm/s). Measurements were first aggregated within
each publication and then across publications so that a publication containing
multiple measurements for a compound did not receive disproportionate weight.
Compounds with a measurement standard deviation greater than 0.3 log10 units
were excluded from the training table. The resulting Public3 dataset contained
5,542 unique compounds.

The ExpansionRX collection was curated independently using the same structure
standardization and identifier-generation procedure. Five compounds in the
ExpansionRX training partition lacked a finite Papp AB label and were excluded,
leaving 2,156 labeled training compounds. The physically separate ExpansionRX
test table contained 1,616 structures, of which 1,615 had a finite Papp AB
label and contributed to the reported metrics. Full-InChIKey comparison found
no exact compound overlap between Public3, the ExpansionRX training partition,
and the ExpansionRX test partition. Combining Public3 and ExpansionRX training
therefore produced a joint dataset of 7,698 unique labeled compounds without
exact test-set leakage.

## Modeling strategies

Three training strategies were evaluated against the same ExpansionRX test
partition:

1. **S1, Public3:** training on the three-source literature collection only,
   representing transfer to ExpansionRX without exposure to ExpansionRX
   training labels.
2. **S2, ExpansionRX:** training on the ExpansionRX training partition only,
   representing an assay-domain-matched model.
3. **S3, Joint:** training on the union of Public3 and ExpansionRX training
   compounds.

Each strategy was evaluated with CheMeleon, Graphormer, and ChemBERTa.
Graphormer was initialized from `graphormer-base-pcqm4mv1`, ChemBERTa from
`DeepChem/ChemBERTa-77M-MLM`, and CheMeleon from its verified pretrained
molecular encoder. The CheMeleon prediction head was trained while the
pretrained message-passing encoder was frozen for the first five epochs, after
which the encoder was unfrozen. Graphormer and ChemBERTa were fine-tuned
end-to-end. Per-device batch sizes were 64, 8, and 32 for CheMeleon,
Graphormer, and ChemBERTa, respectively. Training was run for up to 30 epochs,
and the checkpoint selected by the validation objective was used for external
test prediction.

For every model-strategy combination, four validation designs were examined:
random and scaffold-balanced splitting, each with validation fractions of 0.1
and 0.2. A fixed random seed of 42 was used. This factorial design comprised
12 configurations per model family and 36 configurations in total. The
ExpansionRX test partition was held fixed across configurations.

## Performance and uncertainty analysis

Model performance was evaluated using the coefficient of determination (R2),
root mean squared error (RMSE), mean absolute error (MAE), Pearson correlation,
Spearman rank correlation, and Kendall rank correlation. RMSE was designated
the primary metric because the intended output was a quantitative log10 Papp
prediction. MAE and R2 were treated as supporting measures of absolute
prediction quality, whereas Pearson, Spearman, and Kendall correlations were
used to assess linear association and compound ranking.

Uncertainty in every test metric was estimated by nonparametric bootstrap
sampling of test compounds. For each configuration, 2,000 bootstrap samples
were drawn with replacement at the original test-set size, and each metric was
recomputed. The 2.5th and 97.5th percentiles of the bootstrap distribution were
reported as the 95% confidence interval. Marginal confidence intervals were
used to visualize uncertainty; they were not interpreted as a formal test of
the difference between two models. Comparisons between leading models should
use paired bootstrap differences because all configurations predict the same
test compounds.

## Results

Graphormer produced the strongest quantitative ExpansionRX test performance.
The best configuration used S2 ExpansionRX-only training, a scaffold-balanced
split, and a validation fraction of 0.2. It achieved R2 = 0.223, RMSE = 0.596,
MAE = 0.469, Pearson = 0.551, Spearman = 0.528, and Kendall = 0.371. The closest
competitor was Graphormer trained using S2 with a random split and a validation
fraction of 0.1 (R2 = 0.216, RMSE = 0.599, and MAE = 0.457). The RMSE difference
between these configurations was only 0.0029 log10 units, while the random-split
model had a 0.0122 lower MAE and slightly higher correlations. Consequently,
these two configurations should be regarded as practically close unless a
paired-bootstrap comparison demonstrates a reliable difference. The
scaffold-balanced configuration was retained as the primary quantitative model
because it had the best RMSE and R2 and used a chemically more stringent model-
selection split.

The strongest CheMeleon result was obtained with S2, random splitting, and a
validation fraction of 0.1. This configuration achieved R2 = 0.138,
RMSE = 0.628, MAE = 0.480, Pearson = 0.494, Spearman = 0.514, and
Kendall = 0.356. Its RMSE was approximately 0.032 log10 units higher than that
of the leading Graphormer configuration. The best ChemBERTa configuration in
the RMSE-ranked comparison used S3 joint training with a random 0.2 validation
split and achieved R2 = 0.003, RMSE = 0.675, MAE = 0.522, Pearson = 0.414,
Spearman = 0.403, and Kendall = 0.275. ChemBERTa therefore did not match the
graph-based encoders for this endpoint under the evaluated settings.

Joint training showed a different tradeoff. Graphormer trained on S3 with a
scaffold-balanced 0.2 validation split achieved the strongest observed ranking
statistics: Pearson = 0.596, Spearman = 0.603, and Kendall = 0.425. However,
its R2 decreased to 0.110 and its RMSE increased to 0.638. Relative to the best
S2 scaffold model, joint training improved Pearson, Spearman, and Kendall by
approximately 0.044, 0.075, and 0.054, respectively, but worsened RMSE by
approximately 0.042 log10 units. Thus, the joint model better preserved the
relative ordering of compounds but produced less accurate absolute Papp
values.

### Selected test results

| Model | Strategy | Split | Validation fraction | R2 | RMSE | MAE | Pearson | Spearman | Kendall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graphormer | S2 ExpansionRX | Scaffold | 0.2 | 0.223 | 0.596 | 0.469 | 0.551 | 0.528 | 0.371 |
| Graphormer | S2 ExpansionRX | Random | 0.1 | 0.216 | 0.599 | 0.457 | 0.563 | 0.543 | 0.382 |
| CheMeleon | S2 ExpansionRX | Random | 0.1 | 0.138 | 0.628 | 0.480 | 0.494 | 0.514 | 0.356 |
| Graphormer | S3 Joint | Scaffold | 0.2 | 0.110 | 0.638 | 0.487 | 0.596 | 0.603 | 0.425 |
| CheMeleon | S1 Public3 | Scaffold | 0.2 | 0.103 | 0.640 | 0.507 | 0.433 | 0.428 | 0.290 |
| ChemBERTa | S3 Joint | Random | 0.2 | 0.003 | 0.675 | 0.522 | 0.414 | 0.403 | 0.275 |

## Discussion

The results demonstrate that assay-domain matching was more important for
absolute ExpansionRX permeability prediction than simply increasing the number
of training compounds. ExpansionRX-only training consistently supplied the
leading quantitative models, whereas adding Public3 data did not improve RMSE
or R2. The literature collections were normalized to a common unit and endpoint,
but differences in experimental protocol, laboratory, cell culture, compound
handling, and measurement noise may leave source-dependent offsets or scale
differences that cannot be removed by unit conversion alone. Such residual
heterogeneity provides a plausible explanation for the joint model's stronger
rank correlations but weaker absolute calibration.

Graphormer's advantage suggests that an explicit molecular-graph representation
was beneficial for this Caco-2 endpoint under the evaluated data regime. The
weaker ChemBERTa results should not be interpreted as a general limitation of
SMILES language models, because performance may also depend on tokenizer,
pretraining corpus, optimization schedule, and hyperparameter selection.
Nevertheless, within the controlled comparison performed here, Graphormer gave
the best combination of numerical accuracy and rank association.

The comparison between random and scaffold-balanced validation also illustrates
that stronger internal validation performance does not necessarily identify the
best external model. Random splitting can place close analogs in both training
and validation partitions, yielding an optimistic or less chemically demanding
selection criterion. Although the two leading S2 Graphormer models performed
similarly on the common external set, the scaffold-balanced 0.2 configuration
was preferred for quantitative deployment because it combined the lowest RMSE,
the highest R2, and a more stringent validation design. For a screening workflow
whose primary objective is ranking rather than accurate numerical Papp values,
the S3 scaffold-trained Graphormer is an alternative candidate.

## Limitations and final model-selection policy

The factorial comparison used one random seed per configuration. Repeated seeds
or repeated scaffold splits are needed to quantify variation caused by model
initialization and partition assignment. In addition, because the same
ExpansionRX test set was examined across 36 configurations, it functions as a
common benchmarking set but no longer provides a completely selection-free
estimate for the single chosen model. A final prospective dataset, a second
untouched holdout, or nested/repeated scaffold validation should therefore be
used for the final confirmatory performance claim.

For quantitative Papp prediction, the prespecified selection order is lowest
RMSE, followed by MAE and R2, with scaffold-based validation preferred when
performance is statistically or practically tied. Pearson, Spearman, and
Kendall correlations should be primary only for ranking-oriented applications.
On this basis, the S2 ExpansionRX-only Graphormer selected with a 20%
scaffold-balanced validation set is the recommended quantitative model. The S3
joint Graphormer with the same scaffold validation design is the recommended
candidate when rank ordering is the principal objective.

## Figure caption

**Figure X. Comparison of Caco-2 apparent-permeability models on the common
ExpansionRX test set.** CheMeleon, Graphormer, and ChemBERTa were trained using
Public3 only (S1), ExpansionRX training data only (S2), or their union (S3).
Rows show scaffold-balanced and random validation designs with validation
fractions of 0.1 and 0.2; columns show R2, RMSE, MAE, Pearson, Spearman, and
Kendall metrics. Points denote estimates on the independent ExpansionRX test
partition, and error bars denote molecule-level nonparametric-bootstrap 95%
confidence intervals from 2,000 resamples. Higher values are favorable for R2
and the correlation metrics, whereas lower values are favorable for RMSE and
MAE.
