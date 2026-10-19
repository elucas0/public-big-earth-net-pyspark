#!/usr/bin/env python3
"""
BigEarthNet Data Processing Pipeline
=====================================

This script processes the BigEarthNet satellite imagery dataset, converting raw TIF files
stored in Amazon S3 into Petastorm format suitable for deep learning model training.

The pipeline performs the following steps:
1. Loads metadata containing paths to all image patches
2. Filters data by split (train/validation/test) and samples a percentage
3. Loads Sentinel-2 and Sentinel-1 bands from S3 in parallel using Spark
4. Stacks bands into a 6-channel image array
5. Saves processed data in Petastorm format to S3

Usage:
    spark-submit --master yarn --deploy-mode client data_processing.py --subset-fraction 0.01

Date: January 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import logging
import time
from datetime import datetime

import numpy as np
import rasterio
from io import BytesIO
import s3fs

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import NdarrayCodec

from pyspark.sql import SparkSession


# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 Paths
METADATA_PATH = 's3a://ubs-homes/erasmus/ola/bigearthnet/metadata/metadata_updated.parquet'
BASE_OUTPUT_PATH = 's3a://ubs-homes/erasmus/ola/bigearthnet/peta'
LOG_BASE = 'ubs-homes/erasmus/ola/bigearthnet/logs'

# Spectral Band Configuration
# ---------------------------
# Sentinel-2 bands selected: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
# These bands are sufficient for land cover classification and share the same
# 10-metre native resolution, eliminating the need for spatial resampling.
S2_BANDS = ['B02', 'B03', 'B04', 'B08']

# Sentinel-1 bands selected: VV and VH polarisations
# These radar bands provide complementary information independent of 
# atmospheric conditions and solar illumination.
S1_BANDS = ['VV', 'VH']

# Output image dimensions: 120 x 120 pixels, 6 channels (4 S2 + 2 S1)
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
NUM_CHANNELS = len(S2_BANDS) + len(S1_BANDS)  # 6

# Petastorm Schema Definition
# ---------------------------
# Defines the structure of the output dataset:
# - image: 6-channel stacked satellite imagery (float32)
# - label_map: Land cover classification reference map (float32)
BigEarthNetSchema = Unischema('BigEarthNetSchema', [
    UnischemaField('image', np.float32, (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), NdarrayCodec(), False),
    UnischemaField('label_map', np.float32, (IMAGE_HEIGHT, IMAGE_WIDTH), NdarrayCodec(), False),
])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_version_name(fraction):
    """
    Generate a version name string from the subset fraction.
    
    Args:
        fraction (float): The fraction of data to process (e.g., 0.01 for 1%)
        
    Returns:
        str: Version name string (e.g., '1pct', '5pct', '0.5pct')
    """
    version = f"{int(fraction * 100)}pct"
    if version == "0pct":
        version = f"{fraction * 100:.1f}pct"
    return version


# =============================================================================
# SPARK SESSION
# =============================================================================

def create_spark_session(args, version_name):
    """
    Create and configure a Spark session for distributed processing.
    
    Configures Spark with S3A filesystem support for reading/writing to Amazon S3,
    and sets resource allocation based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments containing memory and executor settings
        version_name (str): Version identifier used in the Spark application name
        
    Returns:
        SparkSession: Configured Spark session ready for use
    """
    builder = (
        SparkSession.builder
        .appName(f"BigEarthNet_{version_name}")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.executor.memory", args.executor_mem)
        .config("spark.driver.memory", args.driver_mem)
        .config("spark.executor.cores", args.executor_cores)
        .config("spark.executor.instances", args.num_executors)
    )
    
    if args.enable_stage_metrics and args.event_log_dir:
        builder = (
            builder
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", args.event_log_dir)
        )
     
    return builder.getOrCreate()


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def process_row(row):
    """
    Process a single image patch from the BigEarthNet dataset.
    
    This function is executed on Spark worker nodes in parallel. It loads
    the required spectral bands from S3, stacks them into a single 6-channel
    image, and packages the result with the corresponding label map.
    
    Band stacking order:
        Channels 0-3: Sentinel-2 bands (B02, B03, B04, B08)
        Channels 4-5: Sentinel-1 bands (VV, VH)
    
    Args:
        row: Spark Row object containing:
            - s2_path: S3 path to Sentinel-2 folder
            - s1_path: S3 path to Sentinel-1 folder
            - ref_path: S3 path to reference map folder
            - patch_id: Unique identifier for the patch
            - s1_name: Sentinel-1 specific filename component
            
    Returns:
        dict: Contains 'image' array (120, 120, 6) and 'label_map' array (120, 120)
    """
    fs = s3fs.S3FileSystem()

    def load_band(s3_path):
        s3_path = s3_path.replace('s3a://', '').replace('s3://', '')
        with fs.open(s3_path, 'rb') as f:
            with rasterio.open(BytesIO(f.read())) as src:
                return src.read(1).astype(np.float32)

    s2 = np.stack([load_band(f"{row.s2_path}{row.patch_id}_{b}.tif") for b in S2_BANDS], axis=-1)
    s1 = np.stack([load_band(f"{row.s1_path}{row.s1_name}_{b}.tif") for b in S1_BANDS], axis=-1)
    label = load_band(f"{row.ref_path}{row.patch_id}_reference_map.tif")
        
    return {"image": np.concatenate([s2, s1], axis=-1), "label_map": label}


def calculate_num_partitions(count, split_name, default_parallelism):
    """
    Calculate optimal number of partitions for a data split.
    
    The partitioning strategy balances parallelism (more partitions = better
    cluster utilisation) against overhead (too many partitions = excessive
    task scheduling cost).
    
    Args:
        count (int): Number of records in the split
        split_name (str): Name of the split ('train', 'validation', 'test')
        default_parallelism (int): Spark's default parallelism setting
        
    Returns:
        int: Number of partitions to use
    """
    # Base calculation: approximately 100 records per partition
    base = max(1, count // 100)
    
    # Ensure at least 4x the default parallelism for good cluster utilisation
    num_partitions = max(default_parallelism * 4, base)
    
    # Double partitions for training data (typically largest split)
    if split_name == "train":
        num_partitions *= 2
    
    return num_partitions


# =============================================================================
# RESULTS AND LOGGING
# =============================================================================

def save_results_to_s3(results, log_path, version_name):
    """
    Save pipeline execution results to S3 as a JSON file.
    
    Args:
        results (dict): Dictionary containing execution metrics
        log_path (str): S3 path prefix for log storage
        version_name (str): Version identifier for the filename
        
    Returns:
        str: Full S3 path where results were saved
    """
    fs = s3fs.S3FileSystem()
    log_file = f'{log_path}/{version_name}.json'
    
    with fs.open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return log_file


def print_pipeline_header(args, version_name):
    """Print pipeline configuration header."""
    print(f"\n{'='*60}")
    print(f"BigEarthNet Data Preparation Pipeline")
    print(f"{'='*60}")
    print(f"Version: {version_name}")
    print(f"Subset fraction: {args.subset_fraction}")
    print(f"Executor memory: {args.executor_mem}")
    print(f"Driver memory: {args.driver_mem}")
    print(f"Executor cores: {args.executor_cores}")
    print(f"Number of executors: {args.num_executors}")
    print(f"{'='*60}\n")


def print_pipeline_summary(total_time, log_file):
    """Print pipeline completion summary."""
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Log saved: s3://{log_file}")
    print(f"{'='*60}\n")


# =============================================================================
# SPLIT PROCESSING
# =============================================================================

def process_split(spark, metadata_df, split_name, args, version_name):
    """
    Process a single data split (train/validation/test).
    
    Args:
        spark: Active Spark session
        metadata_df: DataFrame containing metadata for all patches
        split_name (str): Name of the split to process
        args: Command-line arguments
        version_name (str): Version identifier for output path
        
    Returns:
        dict: Results containing count, time, partitions, and output path
    """
    print(f"\n--- Processing {split_name.upper()} split ---")
    split_start = time.time()
    
    # Filter metadata to current split and sample
    split_df = metadata_df.filter(metadata_df.split == split_name)
    split_df = split_df.sample(fraction=args.subset_fraction, seed=42)
    count = split_df.count()
    
    # Skip empty splits
    if count == 0:
        print(f"Skipped (0 samples)")
        return {"count": 0, "time_sec": 0}
    
    print(f"Records to process: {count:,}")
    
    # Calculate optimal partitioning
    num_partitions = calculate_num_partitions(
        count, 
        split_name, 
        spark.sparkContext.defaultParallelism
    )
    print(f"Using {num_partitions} partitions")
    
    # Define output path
    output_path = f'{BASE_OUTPUT_PATH}/{version_name}/{split_name}'
    print(f"Output path: {output_path}")
    
    # Transform metadata RDD: repartition -> process -> filter nulls -> convert to Spark Rows
    rows_rdd = split_df.rdd \
        .repartition(num_partitions) \
        .map(process_row) \
        .filter(lambda x: x is not None) \
        .map(lambda x: dict_to_spark_row(BigEarthNetSchema, x))
    
    # Write to Petastorm format
    with materialize_dataset(spark, output_path, BigEarthNetSchema, row_group_size_mb=128):
        spark.createDataFrame(rows_rdd, BigEarthNetSchema.as_spark_schema()) \
            .write \
            .mode('overwrite') \
            .parquet(output_path)
    
    # Record timing
    split_time = time.time() - split_start
    print(f"Completed: {count:,} records in {split_time:.2f}s")
    
    return {
        "count": count,
        "time_sec": round(split_time, 2),
        "num_partitions": num_partitions,
        "output_path": output_path
    }


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline(args):
    """
    Execute the complete data processing pipeline.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        dict: Complete results from all splits
    """
    version_name = get_version_name(args.subset_fraction)
    log_path = f"{LOG_BASE}/frac_{args.subset_fraction}/num_executors_{args.num_executors}"
    
    # Print header
    print_pipeline_header(args, version_name)
    
    # Initialise Spark session
    spark = create_spark_session(args, version_name)
    spark.sparkContext.setLogLevel("WARN")
    
    # Start pipeline timer
    pipeline_start = time.time()
    
    # Load metadata
    print("Loading metadata from S3...")
    metadata_df = spark.read.parquet(METADATA_PATH)
    total_records = metadata_df.count()
    print(f"Total records in dataset: {total_records:,}")
    
    # Initialise results dictionary
    results = {
        "version": version_name,
        "subset_fraction": args.subset_fraction,
        "total_records": total_records,
        "executor_mem": args.executor_mem,
        "driver_mem": args.driver_mem,
        "executor_cores": args.executor_cores,
        "num_executors": args.num_executors,
        "splits": {}
    }
    
    # Process each split
    for split_name in ['train', 'validation', 'test']:
        results["splits"][split_name] = process_split(
            spark, metadata_df, split_name, args, version_name
        )
    
    # Calculate total pipeline time
    total_time = time.time() - pipeline_start
    results["total_time_sec"] = round(total_time, 2)
    results["timestamp"] = datetime.now().isoformat()
    
    # Save results to S3
    log_file = save_results_to_s3(results, log_path, version_name)
    
    # Print summary
    print_pipeline_summary(total_time, log_file)
    
    # Stop Spark session
    spark.stop()
    
    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="BigEarthNet Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data selection arguments
    parser.add_argument("--subset-fraction", type=float, default=0.01, help="Fraction of data to process (e.g., 0.01 for 1%%, 0.1 for 10%%)"
)
    
    # Spark resource arguments
    parser.add_argument("--executor-mem", default="16g", help="Memory per executor (default: 16g)")
    parser.add_argument("--driver-mem", default="8g", help="Driver memory (default: 8g)")
    parser.add_argument("--executor-cores", default="3", help="Cores per executor (default: 3)")
    parser.add_argument("--num-executors", default="8", help="Number of executors (default: 8)")
    
    # Monitoring arguments
    parser.add_argument("--enable-stage-metrics", action="store_true", help="Enable Spark event logging")
    parser.add_argument("--event-log-dir", default=None, help="Directory for Spark event logs")
    
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
