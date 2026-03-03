#!/usr/bin/env bash
# =============================================================================
# train.sh — Full model-nano build pipeline
#
# Usage:
#   ./train.sh                  # Run all steps from scratch
#   ./train.sh --from step3     # Resume from a specific step
#   ./train.sh --skip-collect   # Skip data collection (use existing raw data)
#   ./train.sh --synthetic-count 20000
#
# Steps:
#   step1  Clean stale clones
#   step2  Collect real documentation
#   step3  Generate synthetic data
#   step4  Train tokenizer
#   step5  Prepare pretrain dataset
#   step6  Prepare SFT dataset
#   step7  Phase 1 pre-training
#   step8  Phase 2 SFT
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config defaults (override with flags)
# ---------------------------------------------------------------------------
FROM_STEP="step1"
SKIP_COLLECT=false
SYNTHETIC_COUNT=10000
PRETRAIN_EPOCHS=20
SFT_EPOCHS=5
CLONE_DIR="/tmp/model-nano-sources"
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_step() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}${CYAN}  $1${RESET}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

log_ok() {
    echo -e "${GREEN}  ✓ $1${RESET}"
}

log_warn() {
    echo -e "${YELLOW}  ⚠ $1${RESET}"
}

log_err() {
    echo -e "${RED}  ✗ $1${RESET}"
}

log_info() {
    echo -e "  $1"
}

die() {
    log_err "$1"
    exit 1
}

should_run() {
    local step="$1"
    local steps=("step1" "step2" "step3" "step4" "step5" "step6" "step7" "step8")
    local from_idx=-1
    local step_idx=-1
    for i in "${!steps[@]}"; do
        [[ "${steps[$i]}" == "$FROM_STEP" ]] && from_idx=$i
        [[ "${steps[$i]}" == "$step" ]]     && step_idx=$i
    done
    [[ $step_idx -ge $from_idx ]]
}

elapsed() {
    local start=$1
    local end
    end=$(date +%s)
    local secs=$(( end - start ))
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            FROM_STEP="$2"; shift 2 ;;
        --skip-collect)
            SKIP_COLLECT=true; shift ;;
        --synthetic-count)
            SYNTHETIC_COUNT="$2"; shift 2 ;;
        --pretrain-epochs)
            PRETRAIN_EPOCHS="$2"; shift 2 ;;
        --sft-epochs)
            SFT_EPOCHS="$2"; shift 2 ;;
        --clone-dir)
            CLONE_DIR="$2"; shift 2 ;;
        -h|--help)
            grep "^#" "$0" | head -20 | sed 's/^# \?//'
            exit 0 ;;
        *)
            die "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR" data/raw data/sft checkpoints

LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

PIPELINE_START=$(date +%s)

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║        model-nano build pipeline           ║${RESET}"
echo -e "${BOLD}╚════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  Resuming:    $FROM_STEP"
echo -e "  Synthetic:   $SYNTHETIC_COUNT pairs"
echo -e "  Epochs P1:   $PRETRAIN_EPOCHS"
echo -e "  Epochs SFT:  $SFT_EPOCHS"
echo -e "  Log file:    $LOGFILE"

# Verify Python and GPU
python -c "import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'; print(f'  Device:      {gpu}')"

# ---------------------------------------------------------------------------
# STEP 1 — Clean stale clones
# ---------------------------------------------------------------------------
if should_run "step1"; then
    log_step "Step 1/8 — Clean stale clones"
    STEP_START=$(date +%s)

    if [ -d "$CLONE_DIR/git" ]; then
        # Check if man pages actually exist
        if ! ls "$CLONE_DIR/git/Documentation/"*.txt &>/dev/null 2>&1; then
            log_warn "Stale git/git clone detected (no .txt files) — removing"
            rm -rf "$CLONE_DIR/git"
            log_ok "Removed stale clone"
        else
            log_ok "git/git clone looks healthy ($(ls "$CLONE_DIR/git/Documentation/"*.txt 2>/dev/null | wc -l) .txt files)"
        fi
    else
        log_ok "No stale clones found"
    fi

    log_ok "Done ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# STEP 2 — Collect real documentation
# ---------------------------------------------------------------------------
if should_run "step2"; then
    if [ "$SKIP_COLLECT" = true ]; then
        log_warn "Step 2/8 — Skipping data collection (--skip-collect)"
    else
        log_step "Step 2/8 — Collect documentation"
        STEP_START=$(date +%s)

        python data/collect_docs.py --clone-dir "$CLONE_DIR"

        # Verify output
        if [ ! -f "data/raw/docs.jsonl" ]; then
            die "data/raw/docs.jsonl not created"
        fi
        DOC_COUNT=$(wc -l < data/raw/docs.jsonl)
        log_ok "Collected $DOC_COUNT documentation records ($(elapsed $STEP_START))"

        if [ "$DOC_COUNT" -lt 100 ]; then
            log_warn "Low doc count ($DOC_COUNT) — check collect_docs.py output above"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# STEP 3 — Generate synthetic data
# ---------------------------------------------------------------------------
if should_run "step3"; then
    log_step "Step 3/8 — Generate synthetic data"
    STEP_START=$(date +%s)

    python data/generate_synthetic.py --count "$SYNTHETIC_COUNT"

    if [ ! -f "data/raw/synthetic.jsonl" ]; then
        die "data/raw/synthetic.jsonl not created"
    fi
    SYN_COUNT=$(wc -l < data/raw/synthetic.jsonl)
    log_ok "Generated $SYN_COUNT synthetic pairs ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# STEP 4 — Train tokenizer
# ---------------------------------------------------------------------------
if should_run "step4"; then
    log_step "Step 4/8 — Train BPE tokenizer"
    STEP_START=$(date +%s)

    python tokenizer/train_tokenizer.py

    if [ ! -f "tokenizer/tokenizer.json" ]; then
        die "tokenizer/tokenizer.json not created"
    fi
    VOCAB=$(python -c "from tokenizers import Tokenizer; t=Tokenizer.from_file('tokenizer/tokenizer.json'); print(t.get_vocab_size())")
    log_ok "Tokenizer trained — vocab size: $VOCAB ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# STEP 5 — Prepare pretrain dataset
# ---------------------------------------------------------------------------
if should_run "step5"; then
    log_step "Step 5/8 — Prepare pretrain dataset"
    STEP_START=$(date +%s)

    python data/prepare_dataset.py

    if [ ! -f "data/train.bin" ] || [ ! -f "data/val.bin" ]; then
        die "data/train.bin or data/val.bin not created"
    fi
    TRAIN_TOK=$(python -c "import numpy as np; d=np.fromfile('data/train.bin',dtype=np.uint16); print(f'{len(d):,}')")
    log_ok "Pretrain dataset: $TRAIN_TOK train tokens ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# STEP 6 — Prepare SFT dataset
# ---------------------------------------------------------------------------
if should_run "step6"; then
    log_step "Step 6/8 — Prepare SFT dataset (with loss masks)"
    STEP_START=$(date +%s)

    python data/prepare_sft.py

    if [ ! -f "data/sft/train.bin" ] || [ ! -f "data/sft/train.mask.bin" ]; then
        die "data/sft/train.bin or train.mask.bin not created"
    fi
    SFT_SAMPLES=$(python -c "import numpy as np; d=np.fromfile('data/sft/train.bin',dtype=np.uint16); print(len(d)//513)")
    log_ok "SFT dataset: $SFT_SAMPLES training samples ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# STEP 7 — Phase 1: Pre-training
# ---------------------------------------------------------------------------
if should_run "step7"; then
    log_step "Step 7/8 — Phase 1: Pre-training (${PRETRAIN_EPOCHS} epochs)"
    STEP_START=$(date +%s)
    log_info "This is the long step. Monitor GPU: watch -n1 nvidia-smi"
    echo ""

    python -m training.train_pretrain \
        --epochs "$PRETRAIN_EPOCHS" \
        --checkpoint-dir checkpoints/pretrain

    # Find best checkpoint
    if [ -f "checkpoints/pretrain/best.pt" ]; then
        BEST_CKPT="checkpoints/pretrain/best.pt"
    elif [ -f "checkpoints/best.pt" ]; then
        BEST_CKPT="checkpoints/best.pt"
    else
        die "No pretrain checkpoint found after training"
    fi

    log_ok "Pre-training complete — best checkpoint: $BEST_CKPT ($(elapsed $STEP_START))"
    echo "PRETRAIN_BEST=$BEST_CKPT" > .pipeline_state
fi

# ---------------------------------------------------------------------------
# STEP 8 — Phase 2: SFT
# ---------------------------------------------------------------------------
if should_run "step8"; then
    log_step "Step 8/8 — Phase 2: SFT (${SFT_EPOCHS} epochs)"
    STEP_START=$(date +%s)

    # Find best pretrain checkpoint
    if [ -f ".pipeline_state" ]; then
        source .pipeline_state
    fi

    if [ -z "${PRETRAIN_BEST:-}" ]; then
        if [ -f "checkpoints/pretrain/best.pt" ]; then
            PRETRAIN_BEST="checkpoints/pretrain/best.pt"
        elif [ -f "checkpoints/best.pt" ]; then
            PRETRAIN_BEST="checkpoints/best.pt"
        else
            die "No pretrain checkpoint found. Run step7 first or set PRETRAIN_BEST."
        fi
    fi

    log_info "Loading pretrain weights from: $PRETRAIN_BEST"

    python -m training.train_sft \
        --pretrain-checkpoint "$PRETRAIN_BEST" \
        --data-dir data/sft \
        --epochs "$SFT_EPOCHS" \
        --checkpoint-dir checkpoints/sft

    if [ -f "checkpoints/sft/best.pt" ]; then
        SFT_CKPT="checkpoints/sft/best.pt"
    elif [ -f "checkpoints/sft/final.pt" ]; then
        SFT_CKPT="checkpoints/sft/final.pt"
    else
        SFT_CKPT="checkpoints/final.pt"
    fi

    log_ok "SFT complete — checkpoint: $SFT_CKPT ($(elapsed $STEP_START))"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║         Pipeline complete!                 ║${RESET}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Total time:  $(elapsed $PIPELINE_START)"
echo -e "  Log saved:   $LOGFILE"
echo ""
echo -e "  Test your model:"
echo -e "  ${CYAN}git-nano --model checkpoints/sft/best.pt \"undo my last commit\"${RESET}"
echo ""
echo -e "  Run benchmark:"
echo -e "  ${CYAN}python -m eval.benchmark --model checkpoints/sft/best.pt --verbose${RESET}"
echo ""
