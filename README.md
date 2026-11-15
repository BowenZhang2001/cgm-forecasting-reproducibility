# CGM forecasting reproducibility

Code for the CGM forecasting benchmark and the CGMacros multimodal study. The workflow uses the original dataset files, upstream pretrained checkpoints, and the study's final training settings. It writes generated files to a separate work directory.

This repository was developed as part of a collaborative project by the study team.

## Study workflows

| Workflow | Training and evaluation |
| --- | --- |
| Benchmark | Seven datasets, four development settings, and nine 4/12/24 h × 30/60/120 min tasks. The benchmark includes five baselines, four general TSFMs in zero-shot and adapted settings, CGM-JEPA, and compact GluFormer. |
| Image encoders | Fine-tune ConvNeXt-Base on Food-101 for food classification. Initialize a second ConvNeXt-Base from the resulting backbone weights and fine-tune it on Nutrition5k for nutrient prediction. For CGMacros, extract 1,024-dimensional features before the prediction heads of the two frozen encoders. Code for the Food-101 backbone comparison in Supplementary Table S1 is available in the [Food-101 benchmark repository](https://github.com/CocoChengtw/food101-cv-models). |
| Multimodal | Fine-tune Chronos-Bolt-Tiny on all eligible AI-READI training origins. From that checkpoint, adapt the backbone on the CGMacros ID split and independently in each of the 45 LOSO folds. In each split, M0 uses the adapted backbone alone, and the residual variants train on features from the frozen backbone. |
| Evaluation | Export held-out forecasts, error estimates, paired differences, and confidence intervals. Benchmark outputs include clinical metrics and prediabetes/T2D subgroup results. Multimodal outputs include meal-onset, postprandial curve, and glycemic-group results. |

CGM-JEPA checks validation RMSE every 100 updates through step 10,000 and every 500 updates thereafter. The checkpoint with the lowest RMSE across all validation checks is retained. Compact GluFormer trains for 50 epochs with validation every five epochs.

### Final training settings

The launcher uses the final settings from the study. Where forecasting hyperparameters were tuned, validation data were used for selection. Neural forecasting checkpoints were selected by validation RMSE.

| Models | Final settings |
| --- | --- |
| AutoARIMA | [Per-window fitting settings](configs/autoarima.json) |
| Elastic Net | [Regularization settings by development setting and task](configs/elastic_net.csv) |
| LSTM and PatchTST | [Learning rates by development setting and task](configs/neural_baseline_learning_rates.csv) |
| Adapted general TSFMs | [Learning rates by development setting and task](configs/tsfm_learning_rates.csv) |
| CGM-JEPA and compact GluFormer | Both start at a fixed learning rate of `1e-4`. See the [training implementation](src/cgm_pipeline/reproduce_cgm_models.py). |
| CGMacros forecasting | [Backbone adaptation](src/cgm_pipeline/reproduce_multimodal.py) and [residual settings by variant](configs/multimodal.json) |
| Food-101 and Nutrition5k image encoders | [Training settings](configs/image_encoders.json) |

Pretrained checkpoint revisions are pinned in [the general TSFM configuration](configs/tsfm_pretrained.json) and [the CGM model configuration](configs/cgm_pretrained.json).

## Inputs

Download the study releases from their providers. Use the [source-path template](configs/sources.example.json) to point to the extracted files.

- **AI-READI.** Obtain the CGM and metadata matching the study's v3.0.0-compatible release. Place the original `participants.tsv` and glucose `manifest.tsv` alongside the Dexcom G6 JSON tree. The study copy lacked a release marker. The builder verifies the metadata and CGM file inventory against SHA-256 digests pinned in [the AI-READI configuration](configs/datasets/ai_readi.yaml). [AI-READI documentation](https://docs.aireadi.org/docs/3/dataset/healthsheet)
- **Anderson, Brown, and Lynch.** Use each study's original public data export, with the CGM, enrollment and visit/eligibility tables together. Match the file names and checksums in [the Anderson configuration](configs/datasets/anderson.yaml), [the Brown configuration](configs/datasets/brown.yaml), and [the Lynch configuration](configs/datasets/lynch.yaml). [Jaeb public datasets](https://public.jaeb.org/datasets/diabetes)
- **Broll.** Use the `example_data_5_subject` source object in `iglu_4.2.2.tar.gz`. [CRAN version archive](https://cran.r-project.org/src/contrib/Archive/iglu/)
- **Hall.** Use S1 and S5 Data from the Glucotypes paper. Keep the original CGM table and extracted clinical SQLite file under the names specified in [the Hall configuration](configs/datasets/hall.yaml). [Hall et al., PLOS Biology](https://doi.org/10.1371/journal.pbio.2005143)
- **CGMacros.** Obtain files matching the study's PhysioNet v1.0.0-compatible inputs. The study copy lacked a release marker. The builder verifies bio.csv and 45 participant CSVs against the SHA-256 digests pinned in [the CGMacros configuration](configs/datasets/cgmacros.yaml). The multimodal workflow uses the participant photo directories and the study's [photo exclusion list](configs/cgmacros_photo_exclusions.json) by default. The list contains only SHA-256 hashes of the excluded photo files. Matching photos are treated as unavailable. A visual spot check flagged text-only images, obvious leftovers or empty plates, and images of uncertain suitability as pre-meal inputs. [CGMacros on PhysioNet](https://physionet.org/content/cgmacros/1.0.0/)
- **Food-101 and Nutrition5k.** Use the original Food-101 `images` and `meta` directories and the official Nutrition5k RGB splits, dish metadata, overhead RGB images and side-view videos. [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)

The pipeline downloads the pinned public benchmark checkpoints and verifies their checksums.

### Participant exclusions

| Dataset | Baseline eligibility and data quality |
| --- | --- |
| Brown 2019 | A non-automated baseline could not be verified for three adults. One additional adult lacked a randomization date, and another had CGM dates that could not be reliably aligned with the study timeline. |
| Lynch 2022 | Two participants were flagged for prespecified data-quality exclusions. One had already failed baseline eligibility. The additional exclusion reduced the eligible cohort from 146 to 145. |

## Run

1. Set up the environments described in [environments/README.md](environments/README.md).
2. Copy [the source-path template](configs/sources.example.json) and [the run-path template](configs/local_paths.example.json) outside the checkout. Set paths to the raw data, environments, and a new working directory. Use the same extracted CGMacros directory in both files. To override the bundled photo exclusions, add `"image_qc": "/path/to/exclusions.json"` to the run-path file. The JSON file must contain a `rejected_sha256` array.
3. From a Git checkout, run:

   ```bash
   python3 run_study.py --paths /path/to/local_paths.json
   ```

   For separate launches, use `--stage data`, `--stage benchmark`, `--stage image`, `--stage multimodal`, or `--stage metrics`. Benchmark and multimodal stages require the prepared data. The multimodal stage also requires trained image encoders and image features. Full training requires a CUDA-capable GPU. See the [execution environments](environments/README.md) for the reference hardware and software.

   The `metrics` stage checks forecast file coverage before writing CSVs. It requires all benchmark task, model and seed combinations, the multimodal ID inputs, and all 45 LOSO folds. To analyze a supported subset, call the relevant [benchmark](src/cgm_pipeline/reproduce_benchmark_metrics.py) or [multimodal](src/cgm_pipeline/reproduce_multimodal_metrics.py) metrics module directly.

## Outputs

- `benchmark_forecasts/` contains forecasts by development setting, task, model, training seed, and evaluation dataset.
- `image_models/` contains the two selected image checkpoints and their exported CGMacros embeddings.
- `multimodal_forecasts/` contains ID and LOSO forecasts and validation-selected checkpoints.
- `metrics/` contains estimates, paired differences, and pointwise intervals as CSV.

In the benchmark paired CSV, `selected_for_primary` flags comparisons with the zero-shot counterpart and the test-selected baseline. The baseline comparisons are conservative at the point-estimate level within the reference set of LOCF, Elastic Net, LSTM, and PatchTST. These comparisons are exploratory, and their confidence intervals treat the selected reference as fixed.

AutoARIMA metrics use only origins with successful fits. AutoARIMA is outside the paired baseline pool because its evaluable origins can differ from those of the other baselines.

The workflow ends with forecast metrics in CSV format. The manuscript and supplement contain the formatted tables, figures, and computational measurements.
Metric definitions, bootstrap confidence intervals, and Benjamini–Hochberg adjustment are described in the Supplementary Methods (Sections S1-F and S1-G).

## License

The original reproduction code is available under the [MIT License](LICENSE).

The CGM-JEPA and compact GluFormer model definitions and the PatchTST implementation retain their upstream notices and licenses in [src/cgm_pipeline/vendor](src/cgm_pipeline/vendor). Download and use each dataset and checkpoint under its source terms.
