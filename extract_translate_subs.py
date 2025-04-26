import os
import sys
import subprocess
import argparse
import pysubs2
import time
import shutil
import re
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types
from google.genai import types as genai_types
import string # For filename sanitization
import pathlib # For getting script directory
import signal

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    """Signal handler that raises TimeoutException."""
    raise TimeoutException("Chunk translation timed out")

# --- Constants ---
TARGET_LANG_CODE = "es-419"  # Spanish Latin American
TARGET_LANG_NAME = "Spanish Latin American"
SOURCE_LANG_CODE = "eng"
SOURCE_LANG_NAME = "English"
DEFAULT_MODEL = "models/gemini-1.5-flash-latest"
DEFAULT_CHUNK_SIZE = 50
DEFAULT_TIMEOUT = 300 # 5 minutes
RETRY_DELAY = 5 # seconds

# --- Utility Functions ---
def sanitize_filename(filename):
    """Removes or replaces characters potentially problematic for filenames."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub('_+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    return sanitized

# --- FFmpeg Functions ---
def get_subtitle_tracks(input_file):
    """Identifies subtitle tracks in a video file using ffmpeg."""
    print(f"Identifying subtitle tracks in: {os.path.basename(input_file)}")
    command = [ "ffmpeg", "-i", input_file ]
    try:
        process = subprocess.run(command, stderr=subprocess.PIPE, text=True, check=False, encoding='utf-8', errors='ignore')
        output = process.stderr
        subtitle_tracks = {}
        pattern = re.compile(r"Stream #\d+:(\d+)(?:\((\w+)\))?:\s+Subtitle:\s+(\w+)(?:\s+\(title:\s*(.*?)\))?")
        for line in output.splitlines():
            if "Subtitle:" in line:
                match = pattern.search(line.strip())
                if match:
                    stream_index = match.group(1)
                    lang_code = match.group(2) if match.group(2) else "und"
                    codec = match.group(3)
                    title = match.group(4).strip() if match.group(4) else None
                    track_info = {"index": stream_index, "codec": codec, "lang": lang_code, "title": title}
                    subtitle_tracks[stream_index] = track_info
                    print(f"  Found: Index {stream_index}, Lang: {lang_code}, Codec: {codec}, Title: {title}")
        if not subtitle_tracks: print("  No subtitle tracks found.")
        return subtitle_tracks
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg to identify tracks: {e}", file=sys.stderr); print(f"Stderr: {e.stderr}", file=sys.stderr)
        return {}
    except FileNotFoundError: print("Error: ffmpeg not found.", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"Unexpected error identifying tracks: {e}", file=sys.stderr); return {}

def extract_subtitle(input_file, track_index, output_subtitle_file):
    """Extracts a specific subtitle track using its absolute index with ffmpeg."""
    print(f"Extracting subtitle track index {track_index} to: {os.path.basename(output_subtitle_file)}")
    command = [ "ffmpeg", "-y", "-i", input_file, "-map", f"0:{track_index}", "-c:s", "ass", output_subtitle_file ]
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(output_subtitle_file) or os.path.getsize(output_subtitle_file) == 0:
             print(f"  Warning: ffmpeg ran but output file is missing or empty for track {track_index}.", file=sys.stderr)
             if os.path.exists(output_subtitle_file): os.remove(output_subtitle_file)
             return False
        print(f"  Successfully extracted track {track_index}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting subtitle track {track_index}: {e}", file=sys.stderr); print(f"Stderr: {e.stderr}", file=sys.stderr)
        if os.path.exists(output_subtitle_file): os.remove(output_subtitle_file)
        return False
    except FileNotFoundError: print("Error: ffmpeg not found.", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Unexpected error extracting track {track_index}: {e}", file=sys.stderr)
        if os.path.exists(output_subtitle_file): os.remove(output_subtitle_file)
        return False

# --- Translation Functions ---
def translate_chunk(client, model_name, chunk, system_prompt=None, retry_count=0, max_retries=3, timeout=DEFAULT_TIMEOUT):
    """Translates a chunk of subtitle text using the Google Generative AI API, optionally guided by a system prompt."""
    max_delay = RETRY_DELAY * (2 ** max_retries)
    delay = min(RETRY_DELAY * (2 ** retry_count), max_delay)
    full_prompt = ""
    if system_prompt: full_prompt += f"System Prompt:\n{system_prompt}\n\n---\n\n"
    full_prompt += f"""Translate the following {SOURCE_LANG_NAME} subtitle text to {TARGET_LANG_NAME}.
Preserve the original meaning, tone, and context.
Do not add any extra explanations, introductions, or formatting beyond the translated text itself.
Only output the translated text corresponding to the input lines.
Input Text:
---
{chunk}
---
Translated Text ({TARGET_LANG_NAME}):
"""
    response = None # Initialize response
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            print(f"    Sending chunk (length {len(chunk.splitlines())}) to model {model_name} (Attempt {retry_count + 1}). Timeout: {timeout}s")
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                # Use 'config' and 'GenerateContentConfig' as per example
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                    # Other generation config options can be added here if needed
                )
            )
        except TimeoutException:
             print(f"    API call timed out after {timeout} seconds.", file=sys.stderr)
             return "[API Call Timed Out]"
        finally:
            signal.alarm(0) # Disable alarm

        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            translated_text = response.candidates[0].content.parts[0].text.strip()
            print(f"    Received translation chunk.")
            return translated_text
        else:
            finish_reason = response.candidates[0].finish_reason if response and response.candidates else "UNKNOWN"
            safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else []
            print(f"    Warning: Received empty or unexpected response. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}", file=sys.stderr)
            if finish_reason == genai_types.FinishReason.SAFETY: return "[Blocked by Safety Filter]"
            else: return None
    except google_exceptions.GoogleAPIError as e:
        print(f"    API Error translating chunk: {e}", file=sys.stderr)
        if isinstance(e, (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError, google_exceptions.ResourceExhausted)):
            if retry_count < max_retries:
                print(f"    Retrying retriable API error in {delay} seconds...")
                time.sleep(delay)
                return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout)
            else: print(f"    Max retries reached for API error. Skipping chunk.", file=sys.stderr); return None
        else: print(f"    Non-retriable API error. Skipping chunk.", file=sys.stderr); return None
    except Exception as e:
        print(f"    Unexpected error during translation: {type(e).__name__}: {e}", file=sys.stderr)
        if retry_count < max_retries:
             print(f"    Retrying unexpected error in {delay} seconds...")
             time.sleep(delay)
             return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout)
        else: print(f"    Max retries reached for unexpected error. Skipping chunk.", file=sys.stderr); return None

def translate_subtitles_recursive(client, model_name, subs, chunk_size, force_apply_mismatch, current_chunk_size, system_prompt=None, args=None):
    """Translates subtitles, chunking them and handling retries with smaller chunks, optionally guided by a system prompt."""
    # Use the initial chunk_size passed from main for the primary loop step
    initial_chunk_size = chunk_size
    print(f"  Translating subtitles using initial chunk size: {initial_chunk_size}")
    translated_lines = []
    original_lines = [event.text for event in subs]
    total_lines = len(original_lines)

    i = 0 # Current line index
    while i < total_lines:
        # Determine the size of the chunk to *attempt* initially for this iteration
        attempt_chunk_size = min(initial_chunk_size, total_lines - i)
        chunk_original = "\n".join(original_lines[i : i + attempt_chunk_size])
        # original_chunk_len is the length we *expect* to process in this iteration
        original_chunk_len = len(chunk_original.splitlines())

        print(f"  Processing chunk starting at line {i+1} (attempting {original_chunk_len} lines)")

        # Initial translation attempt for the chunk
        # Pass the timeout defined by args.timeout to the API call handler
        current_timeout = args.timeout if args else DEFAULT_TIMEOUT
        translated_chunk = translate_chunk(client, model_name, chunk_original, system_prompt=system_prompt, timeout=current_timeout)

        # --- Local Retry Logic ---
        retry_chunk_size = attempt_chunk_size # Start retry size at the attempted size
        lines_to_add = None # Store the final lines to add for this iteration
        lines_processed_this_iteration = 0 # How many original lines this iteration covers

        while lines_to_add is None: # Loop until we resolve this chunk (success or permanent failure)
            # Check translation result (initial or from previous retry)
            if translated_chunk is None or translated_chunk == "[API Call Timed Out]":
                failure_reason = "timed out" if translated_chunk == "[API Call Timed Out]" else "failed"
                if retry_chunk_size <= 1:
                    print(f"  Error: Translation {failure_reason} for chunk starting at line {i+1} even with chunk size 1. Skipping original chunk.", file=sys.stderr)
                    lines_to_add = ["[Translation Failed]" for _ in range(original_chunk_len)]
                    lines_processed_this_iteration = original_chunk_len # Mark original chunk lines as processed (failed)
                    break # Exit retry loop
                else:
                    # Prepare for retry with smaller size
                    retry_chunk_size = max(1, retry_chunk_size - 5)
                    print(f"  Translation {failure_reason} for chunk starting at line {i+1}. Retrying locally with chunk size {retry_chunk_size}.")

                    # Re-slice the chunk using the smaller retry_chunk_size
                    chunk_original_retry = "\n".join(original_lines[i:min(i + retry_chunk_size, total_lines)])
                    if not chunk_original_retry:
                         print(f"  Error: Retry slicing resulted in empty chunk at index {i}. Skipping original chunk.", file=sys.stderr)
                         lines_to_add = ["[Retry Slicing Error]" for _ in range(original_chunk_len)]
                         lines_processed_this_iteration = original_chunk_len
                         break

                    # Retry the translation with the smaller chunk
                    translated_chunk = translate_chunk(client, model_name, chunk_original_retry, system_prompt=system_prompt, timeout=current_timeout)
                    # Loop will re-evaluate translated_chunk

            elif translated_chunk == "[Blocked by Safety Filter]":
                print(f"  Chunk starting at line {i+1} blocked by safety filter. Adding placeholder.", file=sys.stderr)
                lines_to_add = ["[Blocked by Safety Filter]" for _ in range(original_chunk_len)]
                lines_processed_this_iteration = original_chunk_len # Mark original chunk lines as processed (blocked)
                break # Exit retry loop

            else: # Initial or retry translation was successful text
                chunk_translated_lines = translated_chunk.splitlines()
                # Compare translated length to the length of the *specific chunk that was just translated*
                last_sent_chunk_len = retry_chunk_size # Size of the chunk we *sent* in the last successful API call
                last_translated_chunk_len = len(chunk_translated_lines) # Length of the text returned by API

                if last_translated_chunk_len == last_sent_chunk_len:
                    # Success! The last attempt (initial or retry) worked for the size attempted.
                    print(f"  Successfully translated chunk of size {last_sent_chunk_len} starting at line {i+1}.")
                    lines_to_add = chunk_translated_lines
                    lines_processed_this_iteration = last_sent_chunk_len # Advance by the number of lines successfully processed
                    break # Exit retry loop
                else:
                    # Mismatch occurred on the last attempt
                    print(f"    Warning: Mismatch on attempt with size {last_sent_chunk_len} starting at line {i+1}. Original: {last_sent_chunk_len}, Translated: {last_translated_chunk_len}", file=sys.stderr)
                    if force_apply_mismatch:
                        print("    Applying mismatched translation due to --force-apply-mismatch.")
                        if last_translated_chunk_len < last_sent_chunk_len:
                            chunk_translated_lines.extend(["[Missing Translation]"] * (last_sent_chunk_len - last_translated_chunk_len))
                        else:
                            chunk_translated_lines = chunk_translated_lines[:last_sent_chunk_len]
                        lines_to_add = chunk_translated_lines
                        lines_processed_this_iteration = last_sent_chunk_len # Advance by the number of lines attempted
                        break # Exit retry loop
                    elif retry_chunk_size <= 1: # Check if we were already at the minimum retry size
                         print(f"  Error: Line count mismatch for chunk starting at line {i+1} even with chunk size 1. Skipping original chunk.", file=sys.stderr)
                         lines_to_add = ["[Translation Mismatch]" for _ in range(original_chunk_len)]
                         lines_processed_this_iteration = original_chunk_len # Mark original chunk lines as processed (mismatched)
                         break # Exit retry loop
                    else:
                         # Prepare for retry with smaller size due to mismatch
                         retry_chunk_size = max(1, retry_chunk_size - 5)
                         print(f"  Line count mismatch detected. Retrying locally with chunk size {retry_chunk_size}.")

                         # Re-slice the chunk using the smaller retry_chunk_size
                         chunk_original_retry = "\n".join(original_lines[i:min(i + retry_chunk_size, total_lines)])
                         if not chunk_original_retry:
                              print(f"  Error: Retry slicing resulted in empty chunk at index {i}. Skipping original chunk.", file=sys.stderr)
                              lines_to_add = ["[Retry Slicing Error]" for _ in range(original_chunk_len)]
                              lines_processed_this_iteration = original_chunk_len
                              break

                         # Retry the translation with the smaller chunk
                         translated_chunk = translate_chunk(client, model_name, chunk_original_retry, system_prompt=system_prompt, timeout=current_timeout)
                         # Loop will re-evaluate translated_chunk

        # --- End Local Retry Logic ---

        # Append the final result for this iteration and advance index
        if lines_to_add:
             translated_lines.extend(lines_to_add)
             # processed_lines_count += len(lines_to_add) # Optional: track successful lines added
             i += lines_processed_this_iteration # Advance index by lines processed/skipped
        else:
             # Fallback if retry loop somehow exited without setting lines_to_add
             print(f"  Error: Internal logic error processing chunk at line {i+1}. Skipping original chunk.", file=sys.stderr)
             translated_lines.extend(["[Internal Logic Error]" for _ in range(original_chunk_len)])
             i += original_chunk_len # Advance by original chunk size to avoid infinite loop

    # Final check: Ensure total lines added matches original total lines
    if len(translated_lines) != total_lines:
         print(f"  Warning: Final translated line count ({len(translated_lines)}) differs from original ({total_lines}). Padding/truncating.", file=sys.stderr)
         # Pad or truncate to match original length if absolutely necessary for pysubs2
         if len(translated_lines) < total_lines:
             translated_lines.extend(["[Padding Error]"] * (total_lines - len(translated_lines)))
         else:
             translated_lines = translated_lines[:total_lines]


    # Apply translations back to the SubtitleFile object
    for idx, event in enumerate(subs):
        if idx < len(translated_lines):
             event.text = translated_lines[idx]
        else:
             print(f"  Error: Mismatch applying translation at index {idx}. Subtitle object might be incorrect.", file=sys.stderr)
             break

    print(f"  Finished translation processing for {len(subs)} original lines.")
    return subs

# --- MKVMerge Function ---
def merge_subtitles(input_video_file, translated_subtitle_files, output_video_file):
    """Merges multiple translated subtitle files into the video using mkvmerge."""
    if not translated_subtitle_files:
        print("  No translated subtitles provided for merging. Skipping merge step.", file=sys.stderr)
        return False

    print(f"Merging {len(translated_subtitle_files)} translated subtitle track(s) into: {os.path.basename(output_video_file)}")
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    command = [ "mkvmerge", "-o", output_video_file, input_video_file ] # Source video first

    for sub_file in translated_subtitle_files:
        track_name_suffix = ""
        match = re.search(r'\.track_(\d+)\.', sub_file)
        if match: track_name_suffix = f" (Track {match.group(1)})"
        command.extend([
            "--language", f"0:{TARGET_LANG_CODE}",
            "--track-name", f"0:{TARGET_LANG_NAME}", # Use only the target language name
            "--default-track-flag", "0:no",
            sub_file
        ])

    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        print(f"  Successfully merged subtitles into {os.path.basename(output_video_file)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging subtitles: {e}", file=sys.stderr); print(f"Stderr: {e.stderr}", file=sys.stderr)
        if os.path.exists(output_video_file):
            try: os.remove(output_video_file)
            except OSError as rm_err: print(f"  Failed remove partially merged file: {rm_err}", file=sys.stderr)
        return False
    except FileNotFoundError: print("Error: mkvmerge not found.", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"Unexpected error merging subtitles: {e}", file=sys.stderr); return False

# --- Main Function ---
def main():
    """Main function to parse arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(description=f"Extracts ALL {SOURCE_LANG_NAME} subtitles from video files, translates them to {TARGET_LANG_NAME} using Google Generative AI, and merges them back.")

    parser.add_argument("--api-key", help="Google Generative AI API Key.")
    parser.add_argument("--directory", default=".", help="Directory containing video files.")
    parser.add_argument("--output-dir", default="merged_output", help="Directory for merged videos.")
    parser.add_argument("--cache-dir", default="subtrans_cache", help="Directory for temp subs.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before starting.")
    parser.add_argument("--list-models", action="store_true", help="List compatible models and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model for translation (default: {DEFAULT_MODEL}).")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Lines per API request (default: {DEFAULT_CHUNK_SIZE}).")
    parser.add_argument("--force-apply-mismatch", action="store_true", help="Apply translation even if line counts mismatch.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout per translation API call (default: {DEFAULT_TIMEOUT}s).")
    parser.add_argument("--system-prompt-file", help="Path to system prompt text file.")

    args = parser.parse_args()

    system_prompt = None
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, 'r', encoding='utf-8') as f: system_prompt = f.read().strip()
            if system_prompt: print(f"Using system prompt: {args.system_prompt_file}")
            else: print(f"Warning: System prompt file empty.", file=sys.stderr)
        except Exception as e: print(f"Error reading system prompt file: {e}", file=sys.stderr)

    print("--- Subtitle Extraction and Translation Script ---")

    # Dependency Checks
    if not shutil.which("ffmpeg"): print("Error: ffmpeg not found.", file=sys.stderr); sys.exit(1)
    if not shutil.which("mkvmerge"): print("Error: mkvmerge not found.", file=sys.stderr); sys.exit(1)
    print("Dependencies found.")

    # API Key Handling
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key: print("Error: Google API Key not found.", file=sys.stderr); sys.exit(1)
    # No need to set env var if using Client() correctly
    print("API Key configured.")

    # Initialize Google AI Client
    try:
        # Explicitly pass the API key to the constructor
        client = genai.Client(api_key=api_key)
        print("Google AI Client initialized.")
    except Exception as e: print(f"Error initializing Google AI Client: {e}", file=sys.stderr); sys.exit(1)

    # List Models
    if args.list_models:
        print("\nAvailable Models:"); count=0
        try:
            for model in client.models.list(): print(f"- {model.name}"); count += 1
            if count == 0: print("  No models found.")
            else: print(f"\nFound {count} models.")
        except Exception as e: print(f"Error listing models: {e}", file=sys.stderr)
        sys.exit(0)

    # Directory Setup
    script_dir = pathlib.Path(__file__).parent.resolve()
    input_dir = os.path.abspath(args.directory)
    output_dir = script_dir / args.output_dir
    cache_dir = os.path.abspath(args.cache_dir)
    print(f"Input: {input_dir}\nOutput: {output_dir}\nCache: {cache_dir}")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)

    # Clear Cache
    if args.clear_cache:
        print(f"Clearing cache: {cache_dir}")
        try:
            for item in os.listdir(cache_dir):
                path = os.path.join(cache_dir, item)
                try:
                    if os.path.isfile(path) or os.path.islink(path): os.unlink(path)
                    elif os.path.isdir(path): shutil.rmtree(path)
                except Exception as e: print(f'Failed delete {path}: {e}', file=sys.stderr)
            print("Cache cleared.")
        except Exception as e: print(f"Error clearing cache: {e}", file=sys.stderr)

    # Process Files
    processed_files, skipped_files, failed_files = 0, 0, 0
    try:
        files_to_process = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]
    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(files_to_process)} potential files.")

    for filename in files_to_process:
        input_file_path = os.path.join(input_dir, filename)
        base_name, _ = os.path.splitext(filename)
        sanitized_base_name = sanitize_filename(base_name)
        output_file_path = os.path.join(output_dir, filename)

        print(f"\n--- Processing: {filename} ---")

        if os.path.exists(output_file_path):
            print(f"Output exists. Skipping."); skipped_files += 1; continue

        all_subtitle_tracks = get_subtitle_tracks(input_file_path)
        if not all_subtitle_tracks: print("No subs found. Skipping."); continue

        english_tracks = { idx: info for idx, info in all_subtitle_tracks.items() if info['lang'] == SOURCE_LANG_CODE }
        if not english_tracks: print(f"No {SOURCE_LANG_NAME} tracks. Skipping."); continue

        print(f"Found {len(english_tracks)} {SOURCE_LANG_NAME} track(s): {list(english_tracks.keys())}")
        successfully_translated_files = []
        track_processing_failed = False

        for track_index, track_info in english_tracks.items():
            print(f"\n  -- Track Index: {track_index} --")
            extracted_sub_path = os.path.join(cache_dir, f"{sanitized_base_name}.{SOURCE_LANG_CODE}.track_{track_index}.ass")
            translated_sub_path = os.path.join(cache_dir, f"{sanitized_base_name}.{TARGET_LANG_CODE}.track_{track_index}.ass")

            # Extract
            if os.path.exists(extracted_sub_path): print(f"  Cached source: {os.path.basename(extracted_sub_path)}")
            elif not extract_subtitle(input_file_path, track_index, extracted_sub_path):
                print(f"  Failed extract track {track_index}. Skipping track."); track_processing_failed = True; continue

            # Load
            try:
                subs = pysubs2.load(extracted_sub_path, encoding="utf-8")
                if not subs: print(f"  Warning: Empty sub file. Skipping track.", file=sys.stderr); track_processing_failed = True; continue
                print(f"  Loaded {len(subs)} events.")
            except Exception as e: print(f"  Error loading subs: {e}. Skipping track.", file=sys.stderr); track_processing_failed = True; continue

            # Translate
            if os.path.exists(translated_sub_path):
                 print(f"  Cached translation: {os.path.basename(translated_sub_path)}")
                 successfully_translated_files.append(translated_sub_path)
            else:
                print(f"  Translating {len(subs)} lines...")
                translated_subs = translate_subtitles_recursive(
                    client=client, model_name=args.model, subs=subs, chunk_size=args.chunk_size,
                    force_apply_mismatch=args.force_apply_mismatch,
                    current_chunk_size=args.chunk_size, # Pass initial chunk size
                    system_prompt=system_prompt, args=args
                )
                if translated_subs is None: print("  Translation failed/timed out. Skipping track."); track_processing_failed = True; continue
                # Save
                try:
                    translated_subs.save(translated_sub_path, encoding="utf-8", format="ass")
                    print(f"  Saved translation: {os.path.basename(translated_sub_path)}")
                    successfully_translated_files.append(translated_sub_path)
                except Exception as e:
                    print(f"  Error saving translation: {e}. Skipping track.", file=sys.stderr)
                    if os.path.exists(translated_sub_path):
                        try: os.remove(translated_sub_path)
                        except OSError as rm_err: print(f"    Failed remove corrupted file: {rm_err}", file=sys.stderr)
                    track_processing_failed = True; continue

        # Merge
        if not successfully_translated_files:
            print(f"\nNo tracks translated for {filename}. Skipping merge.")
            if track_processing_failed: failed_files += 1
            continue

        print(f"\nMerging {len(successfully_translated_files)} track(s) for {filename}.")
        if not merge_subtitles(input_file_path, successfully_translated_files, output_file_path):
            print(f"Failed merge for {filename}."); failed_files += 1; continue

        processed_files += 1
        print(f"Successfully processed: {filename}")

    print("\n--- Processing Summary ---")
    print(f"Successfully processed files: {processed_files}")
    print(f"Skipped (already merged): {skipped_files}")
    print(f"Failed (error during processing/merge): {failed_files}")

    print("\nCache cleanup logic placeholder") # Consider implementing actual cleanup

if __name__ == "__main__":
    main()