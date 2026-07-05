#!/bin/bash
# setup_assets.sh — Prepare all required assets (datasets, models, exports)
#
# This script automates the full preparation of the CNN Tutorial Web project:
# 1. Export all datasets to binary format
# 2. Train all model architectures on all datasets
# 3. Export model parameters, layer checkpoints, and torchinfo summaries
# 4. Evaluate test accuracy
#
# Usage:
#   cd /path/to/CNN_Tutorial_web
#   bash setup_assets.sh
#
# Prerequisites:
#   - Python 3.8+
#   - conda with deep-learning environment: conda activate deep-learning
#   - All packages from requirements.txt installed
#
# Device Support:
#   - NVIDIA CUDA (GPU) — automatically detected
#   - Apple Metal Performance Shaders (MPS) — for Apple Silicon Macs
#   - CPU fallback — if no GPU available
#
# Note: For Apple Silicon, ensure PyTorch is installed with MPS support:
#   pip install torch torchvision torchaudio
#
# Output:
#   - public/data/{dataset}/*.bin                 (dataset binaries)
#   - public/models/{dataset}/{arch}/*.onnx       (ONNX layer checkpoints)
#   - public/models/{dataset}/{arch}/model.pth   (full model weights)
#   - public/models/{dataset}/{arch}/torchinfo.json (architecture summary)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="${SCRIPT_DIR}/train"
PUBLIC_DIR="${SCRIPT_DIR}/public"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${BLUE}→ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_section() {
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Activate conda environment
log_step "Activating torch environment..."
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate torch || {
    echo "❌ Failed to activate torch environment"
    echo "Create it with: conda create -n torch python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    exit 1
}

# All datasets to process
DATASETS=("fashion_mnist" "kuzushiji_mnist" "cifar10" "cifar100" "svhn")

# All model architectures to train
MODELS=("linear" "v1" "v1_bn" "v2_small" "v2")

# # ============================================================================
# # STEP 1: Export Datasets
# # ============================================================================
# log_section "STEP 1: Exporting Datasets"

# for dataset in "${DATASETS[@]}"; do
#     log_step "Exporting $dataset dataset..."
#     python "${TRAIN_DIR}/export_dataset.py" --dataset "$dataset" || {
#         echo "⚠️  Failed to export $dataset (this is okay if dataset is unavailable)"
#     }
# done

# log_success "Dataset export complete"

# ============================================================================
# STEP 2: Train Models
# ============================================================================
log_section "STEP 2: Training Models on All Datasets"

total_tasks=$((${#DATASETS[@]} * ${#MODELS[@]}))
completed=0

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        completed=$((completed + 1))
        log_step "[$completed/$total_tasks] Training ${model} on ${dataset}..."

        python "${TRAIN_DIR}/train_${model}.py" --dataset "$dataset" 2>&1 | tail -3 || {
            echo "⚠️  Training ${model} on ${dataset} failed"
            continue
        }
    done
done

log_success "Model training complete"

# ============================================================================
# STEP 3: Export Model Information
# ============================================================================
log_section "STEP 3: Exporting Model Information"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        log_step "Exporting torchinfo for ${model} on ${dataset}..."
        python "${TRAIN_DIR}/export_torchinfo.py" --dataset "$dataset" --model "$model" 2>&1 | grep -E "(Total|Trainable|Mult-Adds)" || true
    done
done

log_success "Torchinfo export complete"

# ============================================================================
# STEP 4: Evaluate Test Accuracy
# ============================================================================
log_section "STEP 4: Evaluating Test Accuracy"

accuracy_log="${SCRIPT_DIR}/model_accuracy.txt"
> "$accuracy_log"  # Clear file

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: $dataset" >> "$accuracy_log"
    for model in "${MODELS[@]}"; do
        log_step "Evaluating ${model} on ${dataset}..."
        python "${TRAIN_DIR}/eval_accuracy.py" --dataset "$dataset" --model "$model" >> "$accuracy_log" 2>&1 || true
    done
    echo "" >> "$accuracy_log"
done

log_success "Accuracy evaluation complete (results saved to model_accuracy.txt)"

# ============================================================================
# STEP 5: Verify Output Structure
# ============================================================================
log_section "STEP 5: Verifying Output Structure"

echo "Dataset binaries:"
ls -lh "${PUBLIC_DIR}"/data/*/test_images.bin 2>/dev/null | head -3 || echo "  (none found yet)"

echo -e "\nModel ONNX files:"
find "${PUBLIC_DIR}"/models -name "*.onnx" -type f 2>/dev/null | head -5 || echo "  (none found yet)"

echo -e "\nTorchinfo summaries:"
find "${PUBLIC_DIR}"/models -name "torchinfo.json" -type f 2>/dev/null | head -5 || echo "  (none found yet)"

# ============================================================================
# Summary
# ============================================================================
log_section "Setup Complete! 🎉"

echo "Next steps:"
echo "  1. Start the dev server:"
echo "     python -m http.server 8080"
echo ""
echo "  2. Open in browser:"
echo "     http://localhost:8080"
echo ""
echo "  3. Navigate to Learn Center → Conv Layer section"
echo "     and load random samples from datasets"
echo ""
echo "Files generated:"
echo "  • public/data/{dataset}/{split}_*.bin      — Dataset binaries"
echo "  • public/models/{dataset}/{arch}/*.onnx    — Layer checkpoints"
echo "  • public/models/{dataset}/{arch}/model.pth — Model weights"
echo "  • model_accuracy.txt                        — Test accuracy for all models"
