# BigEarthNet Scalable Deep Learning

Distributed preprocessing and semantic-segmentation training for the
[BigEarthNet](https://bigearth.net/) satellite-imagery dataset. The project has
two stages:

1. Convert Sentinel-1 and Sentinel-2 imagery from Amazon S3 into Petastorm data
	 using Apache Spark.
2. Train and evaluate a six-channel, 19-class U-Net with TensorFlow/Keras on
	 one or more GPUs.

The pipeline is designed for an AWS EMR/S3 environment. Paths and credentials in
the scripts currently point to the project experiment bucket and may need to be
adapted for another account.

## Requirements

- Python 3.11 or newer
- Java and Apache Spark with `spark-submit` available for preprocessing
- An AWS environment with access to the configured S3 bucket
- TensorFlow 2.17-compatible CPU or CUDA/GPU runtime for training
- `uv` (recommended) or `pip`

The preprocessing job is intended for a Spark/YARN cluster with S3A support. GPU
training can run on a multi-GPU host or a suitable cluster node.

## Installation

Using `uv`:

```bash
uv sync --group cpu   # CPU training
uv sync --group gpu   # GPU training
```

Alternatively, install the pinned environment used for the experiments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure AWS credentials using the standard AWS SDK mechanism, such as an IAM
role, environment variables, or a configured AWS profile. Do not commit secrets
or local credential files.

## Data Layout

The preprocessing script reads metadata from:

```text
s3a://ubs-homes/erasmus/ola/bigearthnet/metadata/metadata_updated.parquet
```

It writes Petastorm datasets below the path configured by `BASE_OUTPUT_PATH` in
`data_processing.py`:

```text
s3a://ubs-homes/erasmus/ola/bigearthnet/peta/<fraction>pct/
├── train/
├── validation/
└── test/
```

Each example contains a `120 x 120 x 6` image and a `120 x 120` label map. The
image channels are, in order, Sentinel-2 `B02`, `B03`, `B04`, `B08`, followed by
Sentinel-1 `VV` and `VH`.

## Preprocessing

Run a small test job first:

```bash
spark-submit --master yarn --deploy-mode client data_processing.py \
	--subset-fraction 0.01
```

For a configured Spark job, specify cluster resources explicitly:

```bash
spark-submit --master yarn --deploy-mode client data_processing.py \
	--subset-fraction 0.05 \
	--executor-mem 16g \
	--driver-mem 8g \
	--executor-cores 3 \
	--num-executors 8 \
	--enable-stage-metrics
```

The main options are:

| Option | Default | Description |
| --- | --- | --- |
| `--subset-fraction` | `0.01` | Fraction of the source dataset to process |
| `--executor-mem` | `16g` | Memory per Spark executor |
| `--driver-mem` | `8g` | Spark driver memory |
| `--executor-cores` | `3` | Cores per executor |
| `--num-executors` | `8` | Number of executors |
| `--enable-stage-metrics` | disabled | Collect SparkMeasure stage metrics |

To run the current scaling matrix of preprocessing experiments:

```bash
chmod +x run_experiments_preprocessing.sh
./run_experiments_preprocessing.sh
```

The script tests 1%, 3%, 5%, and 10% subsets with 1, 4, and 8 executors. Timing
logs are written to the configured S3 logs path.

## Model Training

Training expects the Petastorm datasets to exist in S3. Its current input path is
`s3a://ubs-homes/erasmus/ola/bigearthnet/petastorm/<fraction>pct/`, configured in
`training_model.py`. Align this path with `BASE_OUTPUT_PATH` after preprocessing
if your environment uses the default `.../peta/` output path. A single run looks
like:

```bash
uv run training_model.py \
	--fraction 5 \
	--epochs 5 \
	--strategy one_device \
	--gpus /gpu:0
```

For mirrored multi-GPU training:

```bash
uv run training_model.py \
	--fraction 10 \
	--epochs 10 \
	--strategy mirrored \
	--gpus /gpu:0 /gpu:1 /gpu:2 /gpu:3
```

The training options are:

| Option | Default | Description |
| --- | --- | --- |
| `--fraction` | `5` | Dataset percentage; supported values are 1, 2, 3, 5, and 10 |
| `--epochs` | `5` | Number of training epochs |
| `--strategy` | `one_device` | `one_device` or `mirrored` |
| `--gpus` | none | TensorFlow device names, for example `/gpu:0 /gpu:1` |

To run the predefined training matrix:

```bash
chmod +x run_experiments_training.sh
./run_experiments_training.sh
```

Training uploads timestamped metrics, inference plots, and Keras model files to
the S3 experiment directory configured in `training_model.py`.

## Plotting Metrics

`plot_metrics.py` reads local JSON files from `metrics/` and creates:

- `speedup_scaling_curve.png`
- `accuracy_comparison_curves.png`

Run it after downloading metric JSON files from S3 into `metrics/`:

```bash
python plot_metrics.py
```

The generated plots and local experiment artifacts are ignored by Git. The
repository does not include the dataset, model checkpoints, or experiment
outputs.

## Repository Layout

```text
data_processing.py                  Spark/S3 to Petastorm preprocessing
training_model.py                   U-Net training and evaluation
plot_metrics.py                     Local training-metric plots
plot_results.py                     S3 preprocessing-performance plots
run_experiments_preprocessing.sh    Preprocessing scaling matrix
run_experiments_training.sh         Training scaling matrix
bigearthnet_petastorm_*.ipynb       Exploratory and pipeline notebooks
README_execution.md                 Extended execution notes
```

## Troubleshooting

- **S3 access errors:** verify AWS credentials, IAM permissions, bucket paths,
	and the Spark S3A configuration on the cluster.
- **Java heap errors:** reduce `--subset-fraction`, executor parallelism, or
	the row-group size configured in `data_processing.py`.
- **No GPU detected:** confirm the CUDA-enabled TensorFlow environment and that
	the requested device names exist on the host.
- **Distributed training issues:** use `--strategy one_device` to validate the
	dataset and model on one device before switching to `mirrored`.
- **No local metrics found:** download the training JSON files into `metrics/`
	before running `plot_metrics.py`.

## Generated Files

Experiment outputs are intentionally kept out of version control. Training
metrics are written to `metrics/`, plots and trained Keras models are written to
the repository root, and all of these paths are ignored by Git.