# Execution environments

The reference runs used NVIDIA Tesla T4 GPUs on Linux. Create a separate virtual environment for each row below. Use Python 3.11 for the CGM and forecasting stages and Python 3.10 for the image stage. Run the installation commands with that environment active, then record its Python executable in the run-path configuration.

| Environment in run-path file | Installation |
| --- | --- |
| `data` | `python -m pip install -r environments/data.txt`. |
| `benchmark` | CUDA 12.8 Torch as shown below, then `python -m pip install -r environments/benchmark.txt`. |
| `bolt` | CUDA 12.8 Torch, then `python -m pip install -r environments/bolt.txt`. This environment runs Chronos-Bolt, the CGM-specific models, and the multimodal forecasters. |
| `chronos2` | CUDA 12.8 Torch, then `python -m pip install -r environments/chronos2.txt`. |
| `timesfm` | CUDA 12.8 Torch, then `python -m pip install -r environments/timesfm.txt`. |
| `image` | CUDA 12.1 Torch and TorchVision as shown below, then `python -m pip install -r environments/image.txt`. |

```bash
# Run in each of the benchmark, bolt, chronos2 and timesfm environments.
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Run in the image environment.
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Put the R 4.1.2 `Rscript` executable on `PATH` for Broll ingestion and AutoARIMA. Check the selected executable with `Rscript --version`. AutoARIMA uses `forecast` 8.24.0, `jsonlite`, and `digest`. Set `r_library` to the folder that contains those packages. The installation may use the [archived forecast 8.24.0 source](https://cran.r-project.org/src/contrib/Archive/forecast/forecast_8.24.0.tar.gz).

The launcher imports the source modules from the checkout. For direct module calls, set `PYTHONPATH` to the checkout's `src` directory. `pyproject.toml` describes the data-builder package, and the repository retains `uv.lock` for historical build provenance.
