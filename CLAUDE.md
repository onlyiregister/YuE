# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YuE (乐) is an open-source foundation model for music generation that transforms lyrics into full songs (lyrics2song). It generates complete songs with both vocal and instrumental tracks, supporting diverse genres, languages, and vocal techniques.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n yue python=3.8
conda activate yue

# Install dependencies for inference
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# For finetuning (separate environment)
conda create -n yue-ft python=3.10
conda activate yue-ft
cd finetune/
pip install -r requirements.txt
```

### Inference Commands
```bash
# Basic inference (CoT mode)
cd inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1

# Dual-track ICL mode (with reference audio)
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 \
    [other parameters...]
```

### Finetuning Commands
```bash
# Data preprocessing
cd finetune/
bash scripts/preprocess_data.sh dummy cot $TOKENIZER_MODEL

# Count dataset tokens
bash scripts/count_tokens.sh ./example/mmap/

# Generate training configuration
python core/parse_mixture.py -c example/dummy_data_mixture_cfg.yml

# Run LoRA finetuning
bash scripts/run_finetune.sh
```

### Evaluation Commands
```bash
# Extract pitch values for evaluation
cd evals/pitch_range/
python extract_pitch_values_from_audio/main.py

# Plot vocal range distribution
python plot_violin_plot.py
```

## Architecture Overview

### Core Components

**Two-Stage Generation Pipeline:**
- **Stage 1 (m-a-p/YuE-s1-7B-anneal-*)**: Language model that generates discrete audio codes from text prompts
- **Stage 2 (m-a-p/YuE-s2-1B-general)**: Upsampler that refines audio quality and extends sequences

**Key Modules:**
- `inference/infer.py`: Main inference script with CoT and ICL modes
- `inference/mmtokenizer.py`: Multimodal tokenizer for text and audio
- `inference/codecmanipulator.py`: Audio codec manipulation utilities
- `finetune/scripts/train_lora.py`: LoRA-based finetuning implementation

### Model Variants
- **CoT Models** (`*-cot`): Chain-of-thought mode for diverse generation
- **ICL Models** (`*-icl`): In-context learning mode with reference audio
- **Language Variants**: English (`en`), Chinese (`zh`), Japanese/Korean (`jp-kr`)

### Data Flow
1. **Text Input**: Genre tags + structured lyrics with section labels
2. **Audio Prompt** (optional): Reference audio for style transfer
3. **Stage 1**: Generates discrete audio codes (xcodec format)
4. **Stage 2**: Upsamples and refines to final audio output

## File Structure

```
YuE/
├── inference/           # Inference scripts and models
│   ├── infer.py        # Main inference entry point
│   ├── mmtokenizer.py  # Multimodal tokenization
│   └── xcodec_mini_infer/  # Audio codec models
├── finetune/           # Finetuning framework
│   ├── core/           # Core training modules
│   ├── scripts/        # Training and preprocessing scripts
│   └── example/        # Sample data and configs
├── evals/              # Evaluation scripts
├── prompt_egs/         # Example prompts and audio
└── requirements.txt    # Dependencies
```

## Key Configuration Files

- `finetune/config/ds_config_zero2.json`: DeepSpeed ZeRO-2 configuration
- `finetune/example/dummy_data_mixture_cfg.yml`: Data mixture configuration template
- `inference/xcodec_mini_infer/config.yaml`: Audio codec configuration

## Important Notes

- **GPU Memory**: Requires 24GB+ for full generation, 80GB+ for multiple sessions
- **Flash Attention**: Mandatory for long audio generation to avoid OOM
- **Audio Format**: Uses xcodec with 16kHz sampling rate
- **Context Length**: Default max_new_tokens=3000 (~30s audio per segment)
- **Prompt Structure**: Lyrics must use section labels ([verse], [chorus], etc.) separated by double newlines

## Common Issues

1. **OOM Errors**: Reduce `stage2_batch_size` or `run_n_segments`
2. **Quality Issues**: Try dual-track ICL mode with reference audio
3. **Language Support**: Use appropriate language-specific model variants
4. **Long Generation**: Increase `run_n_segments` for full songs (requires more GPU memory)