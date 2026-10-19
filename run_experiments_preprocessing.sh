#!/bin/bash
#
# BigEarthNet Scaling Experiments
# ================================
#
# This script runs scaling experiments varying both:
# - Data percentage: 1%, 3%, 5%, 10%
# - Number of executors: 1, 4, 8
#
# Each combination produces a separate JSON results file.
#
# Output structure:
#   s3://ubs-homes/erasmus/ola/bigearthnet/logs/
#   ├── frac_0.01/
#   │   ├── num_executors_1/
#   │   ├── num_executors_4/
#   │   └── num_executors_8/
#   ├── frac_0.03/
#   ├── frac_0.05/
#   └── frac_0.10/
#
# Usage:
#   chmod +x bash.sh
#   ./bash.sh
#

echo "========================================"
echo "BigEarthNet Scaling Experiments"
echo "========================================"
echo ""

# Configuration
SCRIPT="data_processing.py"
EVENT_LOG_DIR="s3a://ubs-homes/erasmus/ola/bigearthnet/spark-events/"

# Experiment parameters
PERCENTAGES="0.01 0.03 0.05 0.10"
EXECUTORS="1 4 8"

# Run scaling experiments
echo ""
echo "Running scaling experiments..."
echo "-------------------------------------------"

for pct in $PERCENTAGES; do
    for num_exec in $EXECUTORS; do
        echo ""
        echo "========================================"
        echo "Fraction: $pct | Executors: $num_exec"
        echo "========================================"
        echo ""
        
        spark-submit --master yarn --deploy-mode client $SCRIPT \
            --subset-fraction $pct \
            --num-executors $num_exec \
            --enable-stage-metrics \
            --event-log-dir "$EVENT_LOG_DIR"
        
        echo ""
        echo "Completed: $pct with $num_exec executors"
        echo "Waiting 30 seconds..."
        sleep 30
    done
done

echo ""
echo "========================================"
echo "All scaling experiments completed!"
echo "========================================"
echo ""
echo "Results saved to S3:"
echo "s3://ubs-homes/erasmus/ola/bigearthnet/logs/"
echo ""
echo "Files per experiment:"
echo "  - {pct}.json           : Pipeline timing results"
echo


