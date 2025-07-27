"""
YuE Music Generation Inference Script

This script implements the complete inference pipeline for YuE model, which generates
music from text prompts (lyrics + genre tags) through a two-stage process:
1. Stage 1: Generate discrete audio codes from text using a large language model
2. Stage 2: Upsample and refine the audio codes to high-quality output

The pipeline supports both Chain-of-Thought (CoT) and In-Context Learning (ICL) modes.
"""

import os
import sys
# Add xcodec inference modules to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))

import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf

# YuE-specific modules
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched


# ============================================================================
# Command Line Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description="YuE Music Generation Inference")

# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", 
                   help="Stage 1 model: Generates discrete audio codes from text prompts")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general", 
                   help="Stage 2 model: Upsamples and refines audio codes to 8 quantizer levels")
parser.add_argument("--max_new_tokens", type=int, default=3000, 
                   help="Max tokens per segment (~30s audio). Adjust based on desired segment length")
parser.add_argument("--repetition_penalty", type=float, default=1.1, 
                   help="Controls repetition in audio generation (1.0=no penalty, higher=less repetition)")
parser.add_argument("--run_n_segments", type=int, default=2, 
                   help="Number of lyric segments to generate (more segments = longer song, more GPU memory)")
parser.add_argument("--stage2_batch_size", type=int, default=4, 
                   help="Batch size for Stage 2 inference (reduce if GPU memory is limited)")
# Prompt Configuration:
parser.add_argument("--genre_txt", type=str, required=True, 
                   help="Path to genre tags file (genre, mood, vocal timbre, gender, instruments)")
parser.add_argument("--lyrics_txt", type=str, required=True, 
                   help="Path to structured lyrics file with section labels [verse], [chorus], etc.")

# Audio Prompt (ICL Mode):
parser.add_argument("--use_audio_prompt", action="store_true", 
                   help="Enable single-track ICL mode with reference audio (mix/vocal/instrumental)")
parser.add_argument("--audio_prompt_path", type=str, default="", 
                   help="Path to reference audio file for single-track ICL")
parser.add_argument("--prompt_start_time", type=float, default=0.0, 
                   help="Start time (seconds) for audio prompt extraction")
parser.add_argument("--prompt_end_time", type=float, default=30.0, 
                   help="End time (seconds) for audio prompt extraction (~30s recommended)")
parser.add_argument("--use_dual_tracks_prompt", action="store_true", 
                   help="Enable dual-track ICL mode (better quality, requires separated tracks)")
parser.add_argument("--vocal_track_prompt_path", type=str, default="", 
                   help="Path to separated vocal track for dual-track ICL")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="", 
                   help="Path to separated instrumental track for dual-track ICL")

# Output Configuration:
parser.add_argument("--output_dir", type=str, default="./output", 
                   help="Output directory for all generated files")
parser.add_argument("--keep_intermediate", action="store_true", 
                   help="Keep intermediate Stage 1 and Stage 2 outputs for debugging")
parser.add_argument("--disable_offload_model", action="store_true", 
                   help="Keep Stage 1 model in GPU memory (faster but uses more VRAM)")
parser.add_argument("--cuda_idx", type=int, default=0, help="GPU device index")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

# Audio Codec Configuration:
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', 
                   help='XCodec model configuration file')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', 
                   help='XCodec model checkpoint for audio encoding/decoding')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', 
                   help='Vocos vocoder configuration file')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', 
                   help='Vocos decoder weights for vocal track upsampling')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', 
                   help='Vocos decoder weights for instrumental track upsampling')
parser.add_argument('-r', '--rescale', action='store_true', 
                   help='Rescale final audio output to prevent clipping')


# ============================================================================
# Argument Validation and Setup
# ============================================================================

args = parser.parse_args()

# Validate audio prompt arguments
if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("Please provide --audio_prompt_path when using --use_audio_prompt")
if args.use_dual_tracks_prompt and (not args.vocal_track_prompt_path or not args.instrumental_track_prompt_path):
    raise FileNotFoundError("Please provide both --vocal_track_prompt_path and --instrumental_track_prompt_path when using --use_dual_tracks_prompt")

# Extract key parameters
stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens

# Create output directories
stage1_output_dir = os.path.join(args.output_dir, "stage1")  # Discrete audio codes from Stage 1
stage2_output_dir = os.path.join(args.output_dir, "stage2")  # Upsampled codes from Stage 2
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

def seed_everything(seed=42):
    """Set random seeds for reproducible generation across all libraries."""
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

# ============================================================================
# Model and Tokenizer Initialization
# ============================================================================

print("Initializing models and tokenizers...")
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize multimodal tokenizer (handles both text and audio tokens)
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")

# Load Stage 1 model (text-to-audio-codes generation)
print(f"Loading Stage 1 model: {stage1_model}")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model, 
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    attn_implementation="flash_attention_2",  # Requires flash-attn installation for long sequences
    # device_map="auto",  # Uncomment for automatic device mapping
    )
model.to(device)
model.eval()

# Enable PyTorch compilation for faster inference (PyTorch 2.0+)
if torch.__version__ >= "2.0.0":
    model = torch.compile(model)

# Initialize codec manipulation tools
# Stage 1: 1 quantizer level for initial generation
codectool = CodecManipulator("xcodec", 0, 1)  
# Stage 2: 8 quantizer levels for high-quality upsampling
codectool_stage2 = CodecManipulator("xcodec", 0, 8)  

# Load XCodec model for audio encoding/decoding
print("Loading XCodec model for audio processing...")
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

# ============================================================================
# Helper Classes and Functions
# ============================================================================

class BlockTokenRangeProcessor(LogitsProcessor):
    """
    Logits processor that blocks specific token ranges during generation.
    Used to prevent generation of invalid audio tokens.
    """
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        # Set blocked token probabilities to negative infinity
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def load_audio_mono(filepath, sampling_rate=16000):
    """
    Load audio file and convert to mono at specified sampling rate.
    Used for processing reference audio in ICL mode.
    """
    audio, sr = torchaudio.load(filepath)
    # Convert stereo to mono by averaging channels
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample to target sampling rate if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    """
    Encode audio waveform to discrete codes using XCodec.
    target_bw: Target bandwidth (0.5 kbps for Stage 1)
    """
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)  # Add batch dimension if needed
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    # Transpose and convert to int16 for tokenizer compatibility
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes

def split_lyrics(lyrics):
    """
    Parse structured lyrics with section labels like [verse], [chorus], etc.
    Returns list of formatted lyric segments.
    """
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"  # Match [label] followed by content
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics

# ============================================================================
# Stage 1: Text-to-Audio-Codes Generation
# ============================================================================

print("\n=== Starting Stage 1: Text-to-Audio-Codes Generation ===")

stage1_output_set = []  # Will store paths to generated audio code files

# Load input text files
print("Loading input prompts...")
with open(args.genre_txt) as f:
    genres = f.read().strip()
    print(f"Genre tags: {genres}")

with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())
    print(f"Found {len(lyrics)} lyric segments")

# Construct the main prompt with full context
full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics  # Add individual segments for iterative generation

# Generate unique ID for this run
random_id = uuid.uuid4()
output_seq = None

# Generation hyperparameters (optimized for music generation)
top_p = 0.93           # Nucleus sampling threshold
temperature = 1.0      # Sampling temperature  
repetition_penalty = args.repetition_penalty

# Special tokens for segment boundaries
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')

# Determine how many segments to generate (limited by available lyrics)
run_n_segments = min(args.run_n_segments + 1, len(lyrics))
print(f"Will generate {run_n_segments-1} segments")
# Main generation loop - process each lyric segment iteratively
for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    guidance_scale = 1.5 if i <= 1 else 1.2  # Higher guidance for first segments
    
    if i == 0:
        # Skip the full prompt (index 0), start from individual segments
        continue
    
    if i == 1:
        # First segment: setup audio prompt if using ICL mode
        if args.use_dual_tracks_prompt or args.use_audio_prompt:
            if args.use_dual_tracks_prompt:
                # Dual-track ICL: Process separated vocal and instrumental tracks
                print("Processing dual-track audio prompt...")
                vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                instrumental_ids = load_audio_mono(args.instrumental_track_prompt_path)
                
                # Encode both tracks to discrete codes
                vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                
                # Convert to token IDs
                vocals_ids = codectool.npy2ids(vocals_ids[0])
                instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                
                # Interleave vocal and instrumental tokens (required format for dual-track)
                ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
                
                # Extract specified time range (50 tokens per second * 2 tracks)
                audio_prompt_codec = ids_segment_interleaved[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
                audio_prompt_codec = audio_prompt_codec.tolist()
                
            elif args.use_audio_prompt:
                # Single-track ICL: Process mix, vocal, or instrumental track
                print("Processing single-track audio prompt...")
                audio_prompt = load_audio_mono(args.audio_prompt_path)
                raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                
                # Convert to token IDs and extract time range
                code_ids = codectool.npy2ids(raw_codes[0])
                audio_prompt_codec = code_ids[int(args.prompt_start_time * 50): int(args.prompt_end_time * 50)]  # 50 tokens per second
            
            # Format audio prompt with special tokens
            audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
            sentence_ids = mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
            head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
        else:
            # CoT mode: no audio prompt, use only text
            head_id = mmtokenizer.tokenize(prompt_texts[0])
        
        # Construct prompt for first segment
        prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    else:
        # Subsequent segments: continue from previous generation
        prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

    # Convert to tensor and prepare for generation
    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
    input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
    
    # Handle context window overflow (important for long songs)
    max_context = 16384 - max_new_tokens - 1  # Leave space for new tokens
    if input_ids.shape[-1] > max_context:
        print(f'Section {i}: input length {input_ids.shape[-1]} exceeds context {max_context}, truncating to fit.')
        input_ids = input_ids[:, -max_context:]  # Keep most recent tokens
    
    # Generate audio codes for this segment
    print(f"Generating segment {i}/{run_n_segments-1}...")
    with torch.no_grad():
        output_seq = model.generate(
            input_ids=input_ids, 
            max_new_tokens=max_new_tokens,     # ~30s of audio per segment
            min_new_tokens=100,                # Ensure minimum output length
            do_sample=True,                    # Enable sampling for diversity
            top_p=top_p,                       # Nucleus sampling
            temperature=temperature,           # Control randomness
            repetition_penalty=repetition_penalty,  # Reduce repetition
            eos_token_id=mmtokenizer.eoa,     # End-of-audio token
            pad_token_id=mmtokenizer.eoa,     # Padding token
            # Block invalid token ranges to prevent generation errors
            logits_processor=LogitsProcessorList([
                BlockTokenRangeProcessor(0, 32002),      # Block text tokens
                BlockTokenRangeProcessor(32016, 32016)   # Block specific invalid audio tokens
            ]),
            guidance_scale=guidance_scale,     # Classifier-free guidance strength
        )
        
        # Ensure output ends with end-of-audio token
        if output_seq[0][-1].item() != mmtokenizer.eoa:
            tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
            output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
    
    # Accumulate generated tokens for next iteration
    if i > 1:
        # Concatenate: previous output + current prompt + new generation
        raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
    else:
        # First segment: start accumulation
        raw_output = output_seq

# ============================================================================
# Post-process Stage 1 Output: Extract and Save Audio Codes
# ============================================================================

print("Processing Stage 1 output...")

# Extract generated token sequence and validate structure
ids = raw_output[0].cpu().numpy()
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()  # Start-of-audio positions
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()  # End-of-audio positions

# Sanity check: ensure paired audio segments
if len(soa_idx) != len(eoa_idx):
    raise ValueError(f'Mismatched audio segments: {len(soa_idx)} start tokens, {len(eoa_idx)} end tokens')

print(f"Found {len(soa_idx)} audio segments")

# Extract dual-track audio codes (vocal + instrumental)
vocals = []
instrumentals = []

# Skip reference audio segment if using ICL mode
range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0

for i in range(range_begin, len(soa_idx)):
    # Extract audio codes between start and end tokens
    codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
    
    # Remove separator token if present
    if codec_ids[0] == 32016:
        codec_ids = codec_ids[1:]
    
    # Ensure even number of tokens for dual-track format
    codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
    
    # Separate interleaved vocal and instrumental tokens
    # Format: [vocal_1, inst_1, vocal_2, inst_2, ...]
    vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
    vocals.append(vocals_ids)
    
    instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
    instrumentals.append(instrumentals_ids)

# Concatenate all segments into full tracks
vocals = np.concatenate(vocals, axis=1)
instrumentals = np.concatenate(instrumentals, axis=1)

print(f"Generated audio shape - Vocals: {vocals.shape}, Instrumentals: {instrumentals.shape}")

# Save Stage 1 outputs with descriptive filenames
filename_base = f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}"
vocal_save_path = os.path.join(stage1_output_dir, f"{filename_base}_vtrack".replace('.', '@') + '.npy')
inst_save_path = os.path.join(stage1_output_dir, f"{filename_base}_itrack".replace('.', '@') + '.npy')

np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)

stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)

print(f"Stage 1 complete. Saved outputs to:")
print(f"  Vocal: {vocal_save_path}")
print(f"  Instrumental: {inst_save_path}")


# ============================================================================
# Stage 1 Cleanup and Stage 2 Model Loading
# ============================================================================

# Free GPU memory by offloading Stage 1 model (unless disabled)
if not args.disable_offload_model:
    print("Offloading Stage 1 model to free GPU memory...")
    model.cpu()
    del model
    torch.cuda.empty_cache()

# Load Stage 2 model for upsampling (1 quantizer -> 8 quantizers)
print(f"\n=== Starting Stage 2: Audio Code Upsampling ===")
print(f"Loading Stage 2 model: {stage2_model}")

model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)
model_stage2.to(device)
model_stage2.eval()

# Enable compilation for faster inference
if torch.__version__ >= "2.0.0":
    model_stage2 = torch.compile(model_stage2)

def stage2_generate(model, prompt, batch_size=16):
    """
    Stage 2 upsampling: Convert 1-quantizer codes to 8-quantizer codes.
    Uses teacher forcing - generates additional quantizer levels frame by frame.
    
    Args:
        model: Stage 2 model
        prompt: 1-quantizer audio codes from Stage 1
        batch_size: Process multiple 6-second segments in parallel
    """
    # Prepare codec IDs for Stage 2 processing
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)
    
    # Prepare input sequences based on batch processing
    if batch_size > 1:
        # Batch processing: split into 6-second segments (300 frames each)
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300  # 300 frames = 6 seconds
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        # Format: [SOA] [STAGE_1] [codes] [STAGE_2] for each batch item
        prompt_ids = np.concatenate([
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ], axis=1)
    else:
        # Single sequence processing
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    
    # Block invalid token ranges for Stage 2
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),      # Block text and Stage 1 audio tokens
        BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)  # Block tokens beyond vocab
    ])

    # Teacher forcing generation: provide first quantizer, generate remaining 7
    for frames_idx in range(codec_ids.shape[1]):
        # Add current frame's first quantizer code
        cb0 = codec_ids[:, frames_idx:frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        
        # Generate remaining 7 quantizer levels for this frame
        with torch.no_grad():
            stage2_output = model.generate(
                input_ids=prompt_ids, 
                min_new_tokens=7,    # Exactly 7 additional quantizers
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )
        
        # Verify correct number of new tokens generated
        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, \
               f"Expected 7 new tokens, got {stage2_output.shape[1] - prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Extract and format output
    if batch_size > 1:
        # Concatenate batch results
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        # Single sequence result
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output

def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    """
    Process all Stage 1 outputs through Stage 2 upsampling.
    Handles batching and chunking for efficient GPU utilization.
    """
    stage2_result = []
    
    for i in tqdm(range(len(stage1_output_set)), desc="Stage 2 inference"):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        # Skip if already processed
        if os.path.exists(output_filename):
            print(f'{output_filename} already exists, skipping.')
            stage2_result.append(output_filename)
            continue
        
        # Load Stage 1 output (1-quantizer codes)
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Process in 6-second chunks (300 frames at 50fps)
        output_duration = prompt.shape[-1] // 50 // 6 * 6  # Round down to multiple of 6 seconds
        num_batch = output_duration // 6  # Number of 6-second segments
        
        if num_batch <= batch_size:
            # Process all segments in one batch if small enough
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # Process in chunks if too large for single batch
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300  # 300 frames per 6-second segment
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all processed segments
            output = np.concatenate(segments, axis=0)
        
        # Process any remaining frames (< 6 seconds)
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        
        # Convert token IDs back to codec values
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codec values (temporary workaround)
        # TODO: Find better solution that doesn't affect audio quality
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:  # XCodec valid range
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequent
        
        # Save upsampled output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    
    return stage2_result

# Run Stage 2 inference on all Stage 1 outputs
stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
print(f"Stage 2 complete. Generated files: {stage2_result}")

# ============================================================================
# Audio Reconstruction: Convert Codes to Audio Waveforms
# ============================================================================

print("\n=== Starting Audio Reconstruction ===")

def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    """Save audio tensor to file with optional rescaling to prevent clipping."""
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    limit = 0.99
    max_val = wav.abs().max()
    if rescale:
        wav = wav * min(limit / max_val, 1)
    else:
        wav = wav.clamp(-limit, limit)
    
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
# Step 1: Decode upsampled codes to 16kHz audio using XCodec
print("Step 1: Decoding audio codes to 16kHz waveforms...")
recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, 'mix')
os.makedirs(recons_mix_dir, exist_ok=True)

tracks = []
for npy in stage2_result:
    print(f"Decoding {os.path.basename(npy)}...")
    codec_result = np.load(npy)
    
    # Decode using XCodec model (8-quantizer codes -> 16kHz audio)
    with torch.no_grad():
        decoded_waveform = codec_model.decode(
            torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
    
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
    tracks.append(save_path)
    save_audio(decoded_waveform, save_path, 16000)

print(f"Decoded {len(tracks)} audio tracks to 16kHz")

# Step 2: Mix vocal and instrumental tracks
print("Step 2: Mixing vocal and instrumental tracks...")
for inst_path in tracks:
    try:
        if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) and '_itrack' in inst_path:
            # Find corresponding vocal track
            vocal_path = inst_path.replace('_itrack', '_vtrack')
            if not os.path.exists(vocal_path):
                print(f"Warning: No matching vocal track for {inst_path}")
                continue
            
            # Load and mix tracks
            recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
            vocal_stem, sr = sf.read(vocal_path)
            instrumental_stem, _ = sf.read(inst_path)
            mix_stem = (vocal_stem + instrumental_stem)  # Simple addition mix
            sf.write(recons_mix, mix_stem, sr)
            print(f"Created mix: {recons_mix}")
            
    except Exception as e:
        print(f"Error mixing tracks: {e}")

# Step 3: Upsample to high-quality audio using Vocos vocoder
print("Step 3: Upsampling to high-quality audio using Vocos vocoder...")
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)

# Process each track through specialized vocoders
for npy in stage2_result:
    if '_itrack' in npy:
        # Process instrumental track with instrumental-specific vocoder
        print(f"Upsampling instrumental: {os.path.basename(npy)}")
        instrumental_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'itrack.mp3'),
            args.rescale,
            args,
            inst_decoder,
            codec_model
        )
    else:
        # Process vocal track with vocal-specific vocoder
        print(f"Upsampling vocal: {os.path.basename(npy)}")
        vocal_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
            args.rescale,
            args,
            vocal_decoder,
            codec_model
        )

# Step 4: Mix high-quality vocal and instrumental tracks
print("Step 4: Creating final high-quality mix...")
try:
    mix_output = instrumental_output + vocal_output
    vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
    save_audio(mix_output, vocoder_mix, 44100, args.rescale)  # 44.1kHz output
    print(f"Created high-quality mix: {vocoder_mix}")
except RuntimeError as e:
    print(f"Error creating high-quality mix: {e}")
    print(f"Instrumental shape: {instrumental_output.shape}, Vocal shape: {vocal_output.shape}")

# Step 5: Final post-processing - combine low and high frequency content
print("Step 5: Final post-processing...")
try:
    final_output = os.path.join(args.output_dir, os.path.basename(recons_mix))
    replace_low_freq_with_energy_matched(
        a_file=recons_mix,      # 16kHz reference (low freq)
        b_file=vocoder_mix,     # 44.1kHz upsampled (high freq)
        c_file=final_output,    # Final combined output
        cutoff_freq=5500.0      # Crossover frequency
    )
    print(f"Final output saved: {final_output}")
    print(f"\n🎵 Music generation complete! Check your output directory: {args.output_dir}")
except Exception as e:
    print(f"Error in final post-processing: {e}")
    print(f"You can still find outputs in {vocoder_mix}")
