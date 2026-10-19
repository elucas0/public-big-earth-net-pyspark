import numpy as np
import s3fs
import os
import json
import logging
import argparse
import time

import tensorflow as tf
tf.config.optimizer.set_jit(False)  # Disable XLA for compatibility
import keras as keras

from petastorm.tf_utils import make_petastorm_dataset
from petastorm.codecs import NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField
from petastorm import make_reader

import matplotlib.pyplot as plt

# S3 paths
BUCKET = 'ubs-homes'

# Define schema to read the Petastorm dataset stored on S3
BigEarthNetSchema = Unischema('BigEarthNetSchema', [
    # Multi-spectral Sentinel-2 image (6 bands, 120x120 pixels)
    UnischemaField('image', np.float32, (120, 120, 6), NdarrayCodec(), False),
    UnischemaField('label_map', np.float32, (120, 120), NdarrayCodec(), False),
])

# S3 Client filesystem interface
fs = s3fs.S3FileSystem()

# configuring log
logger = logging.getLogger(__name__)

# CLC → BigEarthNet lookup table
def get_clc_lookup_table():
    """Maps CLC codes to BigEarthNet-19 class indices using a lookup table.

    The original CLC codes range  from 111 to 523
        in the lookup table:
        - index = CLC code
        - value = BigEarthNet class index (0–18)
        - unmapped values default to 19 (background / unknown)

    Returns:
        tf.Tensor: Lookup table as a TensorFlow constant
    """
    # Initialize with 19 (the "Background/Unknown" index)
    # CLC codes are between 111 and 523
    lut = np.full(1000, 19, dtype=np.int32)
    
    # Mapping based on BigEarthNet-19 nomenclature
    mapping = {
        # Urban fabric
        111: 0, 112: 0,
        # Industrial or commercial units
        121: 1,
        # Arable land
        211: 2, 212: 2, 213: 2,
        # Permanent crops
        221: 3, 222: 3, 223: 3,
        # Pastures
        231: 4,
        # Complex cultivation patterns
        242: 5,
        # Land principally occupied by agriculture...
        243: 6,
        # Agro-forestry areas
        244: 7,
        # Broad-leaved forest
        311: 8,
        # Coniferous forest
        312: 9,
        # Mixed forest
        313: 10,
        # Natural grassland and sparsely vegetated areas
        321: 11, 333: 11,
        # Moors, heathland and sclerophyllous vegetation
        322: 12, 323: 12,
        # Transitional woodland, shrub
        324: 13,
        # Beaches, dunes, sands
        331: 14,
        # Inland wetlands
        411: 15, 412: 15,
        # Coastal wetlands
        421: 16, 422: 16,
        # Inland waters
        511: 17, 512: 17,
        # Marine waters
        523: 18
    }
    # Fill lookup table
    for clc_code, class_idx in mapping.items():
        lut[clc_code] = class_idx
    
    # Convert to TensorFlow constant for efficient graph execution
    return tf.constant(lut)

# Global lookup table used during preprocessing
CLC_LOOKUP = get_clc_lookup_table()

# Data preprocessing function
def preprocess_for_model(row):
    """Preprocess a single row for model input

    Args:
        row (petastorm.unischema.BigEarthNetSchema_view): Input row from Petastorm dataset

    Returns:
        tuple: (image, one-hot encoded label)
    """

    # Normalize image to [0, 1] for better training stability
    image = tf.cast(row.image, tf.float32)
    
    # Map CLC codes to 0-18 indices using LUT
    clc_mask = tf.cast(row.label_map, tf.int16)
    indices = tf.gather(CLC_LOOKUP, clc_mask)
    
    # One-hot encoding
    label_one_hot = tf.one_hot(indices, depth=19)
    
    return image, label_one_hot

# U-Net model definition

def build_unet(input_shape=(120, 120, 6)):
    """
    Builds a lightweight U-Net architecture for semantic segmentation.

    Design choices:
      - Shallow network to avoid overfitting
      - Small spatial inputs (120x120)
      - Dropout in bottleneck for regularization

    Args:
        input_shape (tuple): Shape of the input image

    Returns:
        tf.keras.Model: Compiled U-Net model
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    # Level 1: 120x120 -> 64 filters
    c1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = keras.layers.BatchNormalization()(c1)
    p1 = keras.layers.MaxPooling2D(2)(c1) # 60x60

    # Level 2: 60x60 -> 64 filters
    c2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = keras.layers.BatchNormalization()(c2)
    p2 = keras.layers.MaxPooling2D(2)(c2) # 30x30

    # Bottleneck
    b1 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b1 = keras.layers.Dropout(0.3)(b1) # Added dropout to prevent overfitting

    # Decoder
    # Level 2: 30x30 -> 60x60
    u2 = keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(b1)
    # Skip connection between encoder and decoder
    u2 = keras.layers.concatenate([u2, c2])
    c3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    # Level 1: 60x60 -> 120x120
    u1 = keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    # Skip connection between encoder and decoder
    u1 = keras.layers.concatenate([u1, c1])
    c4 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    # OUTPUT
    outputs = keras.layers.Conv2D(19, 1, activation='softmax')(c4)
    
    return keras.Model(inputs, outputs)

# Save model in S3
def save_model_to_s3_fs(model, fs, s3_path):
    """Saves a Keras model to S3 using s3fs.

    Args:
        model (tf.keras.Model): Keras model to save
        fs (s3fs.S3FileSystem): s3fs filesystem object
        s3_path (str): S3 path to save the model
    """
    local_tmp = "temp_model.keras"
    try:
        #  save model locally
        model.save(local_tmp)
        # save model in s3
        # s3fs put: local_path -> s3_path
        fs.put(local_tmp, s3_path)
        logger.info(f"✅ Model saved to {s3_path}")
    finally:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
            
# Save plots
def save_inference_plots(model, dataset, fraction_name, output_dir):
    """Generates and saves inference plots for a few samples from the dataset.

    Args:
        model (tf.keras.Model): Trained model for inference
        dataset (tf.data.Dataset): Dataset for inference
        fraction_name (str): Fraction name for logging
        output_dir (str): Directory to save the plots
    """
    logger.info(f"Generating inference plots for {fraction_name}...")
    # Take only one batch for visualization
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        # Plot first three samples
        for i in range(3):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # Sentinel-2 RGB visualization (bands 2,3,4 scaled)
            ax[0].imshow(np.clip(images[i,:,:,1:4].numpy() * 10, 0, 1))
            ax[0].set_title("S2 RGB")

            # Ground truth labels
            ax[1].imshow(tf.argmax(masks[i], axis=-1), cmap='tab20')
            ax[1].set_title("Ground Truth")

            # Model prediction
            ax[2].imshow(tf.argmax(preds[i], axis=-1), cmap='tab20')
            ax[2].set_title("Prediction")
            
            # Save plot in S3
            plot_path = f"{output_dir}/plots/sample_{i}.png"
            with fs.open(plot_path, 'wb') as f:
                plt.savefig(f, format='png')
            plt.close(fig)


# # Distribution strategy selection
# in our experiments, one_device was used when 1 GPU was selected, and
# mirrored was used with 2 or more GPUs.

def choose_strategy(strategy_name, gpus, batch_size_per_replica=32):
    """Initializes the distribution strategy and calculates the global batch size.

    Args:
        strategy_name (str): Name of the strategy ('one_device', 'mirrored', etc.)
        batch_size_per_replica (int, optional): Batch size per replica. Defaults to 32.

    Raises:
        ValueError: If an unknown strategy is provided.

    Returns:
        tuple: (strategy, global_batch_size)
    """
    if strategy_name == 'one_device':
        # Single-GPU or CPU training
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif strategy_name == 'mirrored' :
        # Multi-GPU synchronous training
        if not gpus or len(gpus) < 1:
            raise ValueError("At least one GPU must be specified for MirroredStrategy.")
        strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Global batch size = batch per GPU * number of GPUs
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    
    return strategy, global_batch_size

# Distributed dataset preparation
def setup_distributed_dataset(reader, batch_size):
    """Gets a distributed dataset from a Petastorm reader.

    Args:
        reader (petastorm.Reader): Petastorm reader
        batch_size (int): Global batch size

    Returns:
        tf.data.Dataset: Prepared dataset for distributed training
    """
    dataset = make_petastorm_dataset(reader).map(preprocess_for_model)
    
    # Set Auto Sharding Policy for distributed training
    # Required because Petastorm is not natively shard-aware
    options = tf.data.Options()
    # Force DATA-based sharding because Petastorm is a custom reader
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    
    # Shuffle, Batch with Global Size, and Prefetch
    dataset = dataset.shuffle(100).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset

def main(args):
    """Main function to run the training and evaluation.

    This function:
      - Parses experiment parameters
      - Sets up paths and logging
      - Initializes the distributed training strategy
      - Trains the U-Net model
      - Evaluates it on a test set
      - Saves metrics, plots, and the trained model to S3

    Args:
        args (argparse.Namespace): Command-line arguments
    """

    # Read experiment configuration
    current_fraction = args.fraction
    epochs = args.epochs
    strategy_name = args.strategy
    gpus_input = args.gpus
    current_time = time.strftime("%Y%m%d-%H%M%S")
    
    # Input dataset paths (S3)

    # Base path where Petastorm Parquet datasets are stored
    base_s3_path = f'ubs-homes/erasmus/ola/bigearthnet/petastorm/{current_fraction}pct'

    # Training and test splits
    train_path = f's3a://{base_s3_path}/train'
    test_path = f's3a://{base_s3_path}/test'

    # Outputs paths

    # Directory for all experiment artifacts
    output_dir = f'ubs-homes/erasmus/elouann/experiments/{current_fraction}/{strategy_name}'
    
    # Model filename with dataset fraction and timestamp
    model_name = f'unet_bigearth_{current_fraction}pct_{current_time}.keras'
    
    # Logging and timing
    overall_start = time.time()
    logger.info(f"========Starting experiment for {current_fraction}%========")
    logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # TensorFlow environment setup
    # Clear previous TensorFlow graphs and variables to avoid memory leaks
    keras.backend.clear_session()

    # Check if any GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Reset GPU memory statistics for cleaner monitoring
        tf.config.experimental.reset_memory_stats('GPU:0')
        logger.info(f"{len(gpus)} Physical GPUs")
    else:
        logger.info("No GPU detected. Training will proceed on CPU.")
    
    # Initialize distribution strategy and scale batch size accordingly
    strategy, global_batch_size = choose_strategy(strategy_name, gpus_input)    
    logger.info(f"Using strategy: {strategy_name} with global batch size: {global_batch_size}")
    
    # --- TRAINING  ---

    # Read training data from S3 using Petastorm
    with make_reader(train_path) as train_reader:
        # Convert Petastorm reader to a distributed tf.data.Dataset
        train_ds = setup_distributed_dataset(train_reader, global_batch_size) 
        
        # Use the strategy scope for model creation and training, this ensures variables are created on the correct devices and mirrored properly
        with strategy.scope():
            # Instantiate and compile the model
            model = build_unet()
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            
            # Measure training time
            t0 = time.time()
            history = model.fit(train_ds, epochs=epochs)
            train_duration = time.time() - t0
            
            # Training performance metrics
            # Throughput calculations
            steps_per_epoch = len(history.history['loss'])
            # Estimate total samples processed
            train_samples_processed = steps_per_epoch * global_batch_size
            # Average training time per sample (in seconds)
            train_time_per_sample = train_duration / (train_samples_processed * epochs)

    # --- EVALUATION ---
    # Read test data from S3
    with make_reader(test_path) as test_reader:

        # Prepare distributed test dataset
        test_ds = setup_distributed_dataset(test_reader, global_batch_size)

        with strategy.scope():
            # Measure evaluation time
            t0 = time.time()
            eval_results = model.evaluate(test_ds)
            eval_duration = time.time() - t0
            
            # Save inference plots to the output directory, for qualitative assessment
            save_inference_plots(model, test_ds, current_fraction, output_dir)

    # --- METRICS PREPARATION ---
    metrics_dict = {
        "fraction": current_fraction,
        "strategy": strategy_name,
        "gpus_used": gpus_input,
        "epochs": epochs,
        "test_metrics": dict(zip(model.metrics_names, eval_results)),
        "training_history": history.history,
        "performance": {
            "total_train_time_sec": train_duration,
            "train_samples_count": train_samples_processed,
            "train_ms_per_sample": train_time_per_sample * 1000,
            "total_test_time_sec": eval_duration,
            "total_experiment_time_sec": time.time() - overall_start
        }
    }

    # --- SAVE ARTIFACTS TO S3
    logger.info("========Uploading artifacts to S3========")
    
    # Save metrics as a JSON file for later analysis
    with fs.open(f"{output_dir}/metrics_{current_time}_{len(gpus_input)}gpus.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    # Save trained model
    save_model_to_s3_fs(model, fs, f"{output_dir}/{model_name}")
    
    logger.info(f"========Finished experiment for {current_fraction}%========")

# Script entry point
if __name__ == "__main__":
    # Define command-line interface
    parser = argparse.ArgumentParser(description="Train U-Net on BigEarthNet with Petastorm and S3")
    # Dataset fraction to use
    parser.add_argument('--fraction', type=int, default=5, choices=[1, 2, 3, 5, 10], help='Fraction of the dataset to use (1, 2, 3, 5, 10)')
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    # Training strategy
    parser.add_argument('--strategy', type=str, default='one_device', choices=['one_device', 'mirrored'], help='Distribution strategy to use')
    # GPU device identifiers for multi-GPU training
    parser.add_argument('--gpus', nargs='+', help='List of GPU device strings, e.g. /gpu:0 /gpu:1 for MirroredStrategy')
    # Parse arguments and launch experiment
    args = parser.parse_args()
    main(args)