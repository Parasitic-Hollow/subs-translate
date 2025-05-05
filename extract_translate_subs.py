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
import concurrent.futures
from functools import partial

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
def translate_chunk(client, model_name, chunk, system_prompt=None, retry_count=0, max_retries=3, timeout=DEFAULT_TIMEOUT, full_srt_context=None):
    """Translates a chunk of subtitle text using the Google Generative AI API, optionally guided by a system prompt and full SRT context."""
    max_delay = RETRY_DELAY * (2 ** max_retries)
    delay = min(RETRY_DELAY * (2 ** retry_count), max_delay)
    full_prompt = ""
    if system_prompt: full_prompt += f"System Prompt:\n{system_prompt}\n\n---\n"
    if full_srt_context:
        print("    Including full SRT context in prompt.") # Added log message
        full_prompt += f"""Full Subtitle Context (SRT Format):
{full_srt_context}
---
"""
    full_prompt += f"""IMPORTANT INSTRUCTION: Preserve any text enclosed in curly braces (e.g., {{{{\\an5\\i1}}}}) exactly as it appears in the original text and in its original position relative to the words. Do NOT translate the content within the curly braces.
CRITICAL: Pay close attention to context (including the full subtitle context provided above if present) to ensure accurate translation, especially regarding gender and number agreement.

Translate the following {SOURCE_LANG_NAME} subtitle text chunk to {TARGET_LANG_NAME}.
Preserve the original meaning, tone, and context.
Do not add any extra explanations, introductions, or formatting beyond the translated text itself.
Only output the translated text corresponding to the input lines.
Input Text Chunk:
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
            # Call generate_content using client.models with the config structure
            response = client.models.generate_content(
                model=f"models/{model_name}", # Specify model name directly here
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0, # Include temperature from previous config
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
                )
                # stream=False # Assuming non-streaming based on previous logic
            )
        except TimeoutException:
             print(f"    API call timed out after {timeout} seconds.", file=sys.stderr)
             return "[API Call Timed Out]"
        finally:
            signal.alarm(0) # Disable alarm

        # Check response (handle potential empty response due to safety or other reasons)
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                translated_text = response.candidates[0].content.parts[0].text.strip()
                print(f"    Received translation chunk.")
                return translated_text
            else:
                # Handle cases where response exists but content is missing (e.g., safety block)
                finish_reason = response.candidates[0].finish_reason if response and response.candidates else "UNKNOWN"
                safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else []
                print(f"    Warning: Received empty or unexpected response structure. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}", file=sys.stderr)
                if finish_reason == genai_types.FinishReason.SAFETY:
                    return "[Blocked by Safety Filter]"
                else:
                    return None # Indicate failure for retry logic
        except (AttributeError, IndexError, TypeError) as resp_err:
             # Catch errors if the response structure is not as expected
             print(f"    Error parsing response structure: {resp_err}. Response: {response}", file=sys.stderr)
             return None # Indicate failure for retry logic

    except google_exceptions.GoogleAPIError as e:
        print(f"    API Error translating chunk: {e}", file=sys.stderr)
        if isinstance(e, (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError, google_exceptions.ResourceExhausted)):
            if retry_count < max_retries:
                print(f"    Retrying retriable API error in {delay} seconds...")
                time.sleep(delay)
                # Recursive call with client, passing context
                return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout, full_srt_context)
            else: print(f"    Max retries reached for API error. Skipping chunk.", file=sys.stderr); return None
        else: print(f"    Non-retriable API error. Skipping chunk.", file=sys.stderr); return None
    except Exception as e:
        print(f"    Unexpected error during translation: {type(e).__name__}: {e}", file=sys.stderr)
        if retry_count < max_retries:
             print(f"    Retrying unexpected error in {delay} seconds...")
             time.sleep(delay)
             # Recursive call with client, passing context
             return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout, full_srt_context)
        else: print(f"    Max retries reached for unexpected error. Skipping chunk.", file=sys.stderr); return None

def translate_subtitles_recursive(client, model_name, subs, chunk_size, force_apply_mismatch, current_chunk_size, system_prompt=None, args=None, fallback_model_name=None, full_srt_context=None):
    """Translates subtitles, chunking them, skipping complex tag lines, and handling retries."""
    initial_chunk_size = chunk_size
    print(f"  Translating subtitles using initial chunk size: {initial_chunk_size}")

    # --- Pre-processing: Identify lines to translate ---
    original_lines = [event.text for event in subs]
    total_original_lines = len(original_lines)
    lines_to_translate_info = [] # List of (original_index, text)
    skipped_line_count = 0
    for idx, line in enumerate(original_lines):
        # --- Reverted Skipping Logic ---
        is_complex_tag = re.search(r'\\t\(', line) or re.search(r'\\p', line)
        if not is_complex_tag:
            lines_to_translate_info.append((idx, line))
        else:
            print(f"  Skipping line {idx+1} because it contains complex tag (\\t( or \\p): {line[:60]}...")
            skipped_line_count += 1
        # --- End Reverted Skipping Logic ---

    translatable_texts = [text for _, text in lines_to_translate_info]
    total_translatable_lines = len(translatable_texts)
    print(f"  Identified {total_translatable_lines} lines for translation (skipped {skipped_line_count}).")

    if total_translatable_lines == 0:
        print("  No lines identified for translation. Returning original subtitles.")
        return subs # Return original subs if nothing to translate

    # Pre-fill results with original text
    final_translated_lines = list(original_lines)
    # This list will store results corresponding *only* to the lines_to_translate_info
    all_translated_results = []

    # --- Main Translation Loop (operates on translatable_texts) ---
    processed_translatable_idx = 0 # Index for translatable_texts
    while processed_translatable_idx < total_translatable_lines:
        # Determine the size of the chunk to *attempt* initially for this iteration
        attempt_chunk_size = min(initial_chunk_size, total_translatable_lines - processed_translatable_idx)
        # Get the actual text chunk to translate
        chunk_to_translate_list = translatable_texts[processed_translatable_idx : processed_translatable_idx + attempt_chunk_size]
        chunk_to_translate = "\n".join(chunk_to_translate_list)
        # expected_chunk_len is the length we *expect* to process in this iteration (of translatable lines)
        expected_chunk_len = len(chunk_to_translate_list)

        print(f"  Processing translatable chunk starting at index {processed_translatable_idx} (attempting {expected_chunk_len} lines)")

        # Initial translation attempt for the chunk
        current_timeout = args.timeout if args else DEFAULT_TIMEOUT
        translated_chunk = translate_chunk(client, model_name, chunk_to_translate, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)

        # --- Local Retry Logic (operates on translatable_texts) ---
        retry_chunk_size = expected_chunk_len # Start retry size at the attempted size
        lines_added_this_iteration = None # Store the final lines to add for this iteration
        lines_processed_this_iteration = 0 # How many *translatable* lines this iteration covers
        fallback_attempts = {} # Track fallback attempts per *translatable* line index

        while lines_added_this_iteration is None: # Loop until we resolve this chunk
            current_translatable_line_index = processed_translatable_idx # For fallback tracking

            if translated_chunk is None or translated_chunk == "[API Call Timed Out]":
                failure_reason = "timed out" if translated_chunk == "[API Call Timed Out]" else "failed"
                if retry_chunk_size <= 1:
                    # --- Single Translatable Line Failure/Timeout Logic ---
                    if fallback_model_name:
                        fallback_count = fallback_attempts.get(current_translatable_line_index, 0)
                        if fallback_count < 3:
                            fallback_attempts[current_translatable_line_index] = fallback_count + 1
                            print(f"  Translation {failure_reason} for single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). Attempting fallback model (try {fallback_count + 1}/3): {fallback_model_name}")
                            # Get the single line text from translatable_texts
                            single_line_text_list = translatable_texts[current_translatable_line_index:min(current_translatable_line_index + 1, total_translatable_lines)]
                            if not single_line_text_list:
                                print(f"  Error: Fallback retry slicing resulted in empty translatable chunk at index {current_translatable_line_index}. Skipping.", file=sys.stderr)
                                lines_added_this_iteration = ["[Fallback Slicing Error]"]
                                lines_processed_this_iteration = 1
                                break # Exit inner while loop
                            single_line_text = "\n".join(single_line_text_list)
                            translated_chunk = translate_chunk(client, fallback_model_name, single_line_text, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                            continue # Re-evaluate the result
                        else:
                            print(f"  Failed single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}) after 3 fallback attempts.", file=sys.stderr)
                            lines_added_this_iteration = ["[Translation Failed]"]
                            lines_processed_this_iteration = 1
                            break
                    else:
                        print(f"  Translation {failure_reason} for single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). No fallback. Skipping line.", file=sys.stderr)
                        lines_added_this_iteration = ["[Translation Failed]"]
                        lines_processed_this_iteration = 1
                        break
                    # --- End Single Translatable Line Failure/Timeout Logic ---
                else:
                    # Prepare for retry with smaller size
                    retry_chunk_size = max(1, retry_chunk_size - 5)
                    print(f"  Translation {failure_reason} for translatable chunk starting at index {processed_translatable_idx}. Retrying locally with chunk size {retry_chunk_size}.")
                    # Re-slice the chunk using the smaller retry_chunk_size from translatable_texts
                    chunk_to_translate_retry_list = translatable_texts[processed_translatable_idx:min(processed_translatable_idx + retry_chunk_size, total_translatable_lines)]
                    if not chunk_to_translate_retry_list:
                         print(f"  Error: Retry slicing resulted in empty translatable chunk at index {processed_translatable_idx}. Skipping original chunk attempt.", file=sys.stderr)
                         lines_added_this_iteration = ["[Retry Slicing Error]" for _ in range(expected_chunk_len)]
                         lines_processed_this_iteration = expected_chunk_len
                         break
                    chunk_to_translate_retry = "\n".join(chunk_to_translate_retry_list)
                    # Retry the translation with the smaller chunk
                    translated_chunk = translate_chunk(client, model_name, chunk_to_translate_retry, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                    # Loop will re-evaluate translated_chunk

            elif translated_chunk == "[Blocked by Safety Filter]":
                print(f"  Translatable chunk starting at index {processed_translatable_idx} blocked by safety filter. Adding placeholder.", file=sys.stderr)
                lines_added_this_iteration = ["[Blocked by Safety Filter]" for _ in range(expected_chunk_len)]
                lines_processed_this_iteration = expected_chunk_len # Mark original chunk lines as processed (blocked)
                break # Exit retry loop

            else: # Initial or retry translation was successful text
                chunk_translated_lines = translated_chunk.splitlines()
                # Compare translated length to the length of the *specific translatable chunk that was just sent*
                last_sent_chunk_len = retry_chunk_size # Size of the chunk we *sent* in the last successful API call
                last_translated_chunk_len = len(chunk_translated_lines) # Length of the text returned by API

                if last_translated_chunk_len == last_sent_chunk_len:
                    # Success! The last attempt (initial or retry) worked for the size attempted.
                    print(f"  Successfully translated translatable chunk of size {last_sent_chunk_len} starting at index {processed_translatable_idx}.")
                    # --- Strip HTML-like tags ---
                    cleaned_lines = []
                    for i, line in enumerate(chunk_translated_lines):
                        if '<' in line and '>' in line:
                            original_line_preview = line[:60] + '...' if len(line) > 60 else line
                            stripped_line = re.sub(r"<[^>]+>", "", line).strip()
                            if stripped_line != line: # Log only if something was actually stripped
                                print(f"    Stripped tags from line {i+1} (original index {lines_to_translate_info[processed_translatable_idx + i][0]+1}): '{original_line_preview}' -> '{stripped_line}'")
                            cleaned_lines.append(stripped_line)
                        else:
                            cleaned_lines.append(line)
                    chunk_translated_lines = cleaned_lines # Replace with cleaned lines
                    # --- End Strip HTML-like tags ---
                    lines_added_this_iteration = chunk_translated_lines
                    lines_processed_this_iteration = last_sent_chunk_len # Advance by the number of lines successfully processed
                    break # Exit retry loop
                elif len(chunk_translated_lines) != last_sent_chunk_len:
                    # Mismatch occurred on the last attempt
                    print(f"    Warning: Mismatch on attempt with size {last_sent_chunk_len} starting at translatable index {processed_translatable_idx}. Original: {last_sent_chunk_len}, Translated: {last_translated_chunk_len}", file=sys.stderr)
                    if force_apply_mismatch:
                        print("    Applying mismatched translation due to --force-apply-mismatch.")
                        if last_translated_chunk_len < last_sent_chunk_len:
                            chunk_translated_lines.extend(["[Missing Translation]"] * (last_sent_chunk_len - last_translated_chunk_len))
                        else: # last_translated_chunk_len >= last_sent_chunk_len
                            chunk_translated_lines = chunk_translated_lines[:last_sent_chunk_len]
                        # --- Strip HTML-like tags (only when force applying) ---
                        cleaned_lines = []
                        for i, line in enumerate(chunk_translated_lines):
                            if '<' in line and '>' in line:
                                original_line_preview = line[:60] + '...' if len(line) > 60 else line
                                stripped_line = re.sub(r"<[^>]+>", "", line).strip()
                                if stripped_line != line: # Log only if something was actually stripped
                                    print(f"    Stripped tags from line {i+1} (original index {lines_to_translate_info[processed_translatable_idx + i][0]+1}): '{original_line_preview}' -> '{stripped_line}'")
                                cleaned_lines.append(stripped_line)
                            else:
                                cleaned_lines.append(line)
                        chunk_translated_lines = cleaned_lines # Replace with cleaned lines
                        # --- End Strip HTML-like tags ---
                        lines_added_this_iteration = chunk_translated_lines
                        lines_processed_this_iteration = last_sent_chunk_len # Advance by the number of lines attempted
                        break # Exit retry loop (force apply)
                    elif retry_chunk_size <= 1: # Mismatch occurred even with size 1
                        # --- Single Translatable Line Mismatch Logic ---
                        current_translatable_line_index = processed_translatable_idx # Index of the single line causing mismatch
                        if fallback_model_name:
                            fallback_count = fallback_attempts.get(current_translatable_line_index, 0)
                            if fallback_count < 3:
                                fallback_attempts[current_translatable_line_index] = fallback_count + 1
                                print(f"    Line count mismatch for single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). Attempting fallback model (try {fallback_count + 1}/3): {fallback_model_name}")
                                single_line_text_list = translatable_texts[current_translatable_line_index:min(current_translatable_line_index + 1, total_translatable_lines)]
                                if not single_line_text_list:
                                    print(f"    Error: Fallback retry slicing resulted in empty translatable chunk at index {current_translatable_line_index}. Skipping.", file=sys.stderr)
                                    lines_added_this_iteration = ["[Fallback Slicing Error]"]
                                    lines_processed_this_iteration = 1
                                    break # Exit inner while loop
                                single_line_text = "\n".join(single_line_text_list)
                                # Use the *original* client and model unless fallback is explicitly different
                                translated_chunk = translate_chunk(client, fallback_model_name, single_line_text, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                                # Don't break or set lines_added/processed here, let the loop re-evaluate translated_chunk
                                continue # Re-evaluate the result with the fallback translation
                            else: # Failed after 3 fallback attempts
                                print(f"    Failed single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}) due to mismatch after 3 fallback attempts.", file=sys.stderr)
                                lines_added_this_iteration = ["[Translation Mismatch]"]
                                lines_processed_this_iteration = 1
                                break # Exit inner while loop (failed fallback)
                        else: # No fallback model configured
                            print(f"    Line count mismatch for single translatable line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). No fallback. Skipping line.", file=sys.stderr)
                            lines_added_this_iteration = ["[Translation Mismatch]"]
                            lines_processed_this_iteration = 1
                            break # Exit inner while loop (no fallback)
                        # --- End Single Translatable Line Mismatch Logic ---
                    else: # Mismatch, but can retry with smaller chunk
                        retry_chunk_size = max(1, retry_chunk_size // 2) # Halve chunk size
                        print(f"    Retrying with smaller chunk size: {retry_chunk_size}")
                        # No break here, continue the inner while loop to retry with the new smaller chunk size
                        # The outer loop will re-slice based on the updated retry_chunk_size
                        chunk_to_translate_retry = "\n".join(chunk_to_translate_retry_list)
                        # Retry the translation with the smaller chunk
                        translated_chunk = translate_chunk(client, model_name, chunk_to_translate_retry, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                        # Loop will re-evaluate translated_chunk

        # --- End Local Retry Logic ---

        # Append the final result for this iteration to all_translated_results and advance index
        if lines_added_this_iteration:
             all_translated_results.extend(lines_added_this_iteration)
             processed_translatable_idx += lines_processed_this_iteration # Advance index by translatable lines processed/skipped
        else:
             # Fallback if retry loop somehow exited without setting lines_added_this_iteration
             print(f"  Error: Internal logic error processing translatable chunk at index {processed_translatable_idx}. Skipping original chunk attempt.", file=sys.stderr)
             all_translated_results.extend(["[Internal Logic Error]" for _ in range(expected_chunk_len)])
             processed_translatable_idx += expected_chunk_len # Advance by original chunk size to avoid infinite loop

        # --- API Delay ---
        if args and args.api_delay > 0:
            print(f"  Applying API delay: {args.api_delay}s")
            time.sleep(args.api_delay)

    # --- Apply Translated Results Back ---
    if len(all_translated_results) != total_translatable_lines:
        print(f"  Warning: Mismatch between expected translatable lines ({total_translatable_lines}) and actual results ({len(all_translated_results)}). This indicates a potential logic error.", file=sys.stderr)
        # Attempt to recover, but this shouldn't happen with correct logic
        if len(all_translated_results) < total_translatable_lines:
            all_translated_results.extend(["[Result Count Mismatch Error]"] * (total_translatable_lines - len(all_translated_results)))
        else:
            all_translated_results = all_translated_results[:total_translatable_lines]

    print(f"  Mapping {len(all_translated_results)} results back to original positions.")
    for i, (original_idx, _) in enumerate(lines_to_translate_info):
        if i < len(all_translated_results):
            final_translated_lines[original_idx] = all_translated_results[i]
        else:
            # This case should be prevented by the check above, but added for safety
            print(f"  Error: Missing result for translatable line with original index {original_idx}. Keeping original.", file=sys.stderr)
            final_translated_lines[original_idx] = "[Missing Result Error]"


    # Apply final combined translations back to the SubtitleFile object
    if len(final_translated_lines) != total_original_lines:
         # This check should ideally not be needed if pre-fill and mapping are correct
         print(f"  FATAL Error: Final line count ({len(final_translated_lines)}) doesn't match original ({total_original_lines}) before applying to subs object.", file=sys.stderr)
         # Avoid modifying the subs object if counts mismatch drastically
         return subs # Or raise an exception

    for idx, event in enumerate(subs):
        if idx < len(final_translated_lines):
             event.text = final_translated_lines[idx]
        else:
             # This should not happen if the length check above passed
             print(f"  Error: Mismatch applying final translation at index {idx}. Subtitle object might be incorrect.", file=sys.stderr)
             break

    print(f"  Finished translation processing for {total_original_lines} original lines ({skipped_line_count} skipped).")
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


# --- Worker Function for Parallel Processing ---
def process_single_file(input_file_path, args, system_prompt, cache_dir, output_dir):
    """Processes a single video file: extract, translate, merge."""
    filename = os.path.basename(input_file_path)
    base_name, _ = os.path.splitext(filename)
    sanitized_base_name = sanitize_filename(base_name)
    output_file_path = os.path.join(output_dir, filename)

    print(f"\n--- Processing: {filename} (PID: {os.getpid()}) ---")

    if os.path.exists(output_file_path):
        print(f"Output exists. Skipping."); return 'skipped'

    # --- Initialize Client within Worker ---
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(f"Error: Google API Key not found for worker (PID: {os.getpid()}). Skipping file.", file=sys.stderr)
        return 'failed' # Indicate failure

    client = None # Initialize client to None
    try:
        # Explicitly pass the API key to the constructor
        client = genai.Client(api_key=api_key)
        # print(f"Google AI Client initialized for worker (PID: {os.getpid()}).") # Optional: verbose logging
    except Exception as e:
        print(f"Error initializing Google AI Client in worker (PID: {os.getpid()}): {e}. Skipping file.", file=sys.stderr)
        return 'failed' # Indicate failure

    # --- File Processing Logic (moved from main loop) ---
    all_subtitle_tracks = get_subtitle_tracks(input_file_path)
    if not all_subtitle_tracks: print("No subs found. Skipping."); return 'skipped' # Treat as skipped if no subs

    english_tracks = { idx: info for idx, info in all_subtitle_tracks.items() if info['lang'] == SOURCE_LANG_CODE }
    if not english_tracks: print(f"No {SOURCE_LANG_NAME} tracks. Skipping."); return 'skipped' # Treat as skipped

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

        # --- Convert extracted ASS to SRT for context ---
        temp_srt_path = extracted_sub_path + ".temp.srt"
        full_srt_context = None
        try:
            print(f"  Converting extracted ASS to SRT for context: {os.path.basename(temp_srt_path)}")
            convert_command = ["ffmpeg", "-y", "-i", extracted_sub_path, temp_srt_path]
            convert_process = subprocess.run(convert_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            if os.path.exists(temp_srt_path) and os.path.getsize(temp_srt_path) > 0:
                print(f"  Successfully converted to SRT.")
                try:
                    with open(temp_srt_path, 'r', encoding='utf-8') as f_srt:
                        full_srt_context = f_srt.read()
                    print(f"  Read {len(full_srt_context.splitlines())} lines from temporary SRT for context.")
                except Exception as read_err:
                    print(f"  Error reading temporary SRT file {temp_srt_path}: {read_err}", file=sys.stderr)
                    full_srt_context = None # Ensure context is None if read fails
            else:
                print(f"  Warning: SRT conversion ran but output file is missing or empty.", file=sys.stderr)
                full_srt_context = None
        except subprocess.CalledProcessError as e:
            print(f"  Error converting ASS to SRT: {e}", file=sys.stderr); print(f"  Stderr: {e.stderr}", file=sys.stderr)
            full_srt_context = None
        except FileNotFoundError:
            print("  Error: ffmpeg not found during SRT conversion.", file=sys.stderr)
            full_srt_context = None # Cannot proceed without ffmpeg
            # Optionally: sys.exit(1) or raise an exception if ffmpeg is critical
        except Exception as e:
            print(f"  Unexpected error during SRT conversion: {e}", file=sys.stderr)
            full_srt_context = None
        finally:
            # Ensure temporary SRT file is deleted
            if os.path.exists(temp_srt_path):
                try:
                    os.remove(temp_srt_path)
                    # print(f"  Deleted temporary SRT file: {os.path.basename(temp_srt_path)}") # Optional: verbose logging
                except OSError as rm_err:
                    print(f"  Warning: Failed to delete temporary SRT file {temp_srt_path}: {rm_err}", file=sys.stderr)

        # --- Load Original ASS for Translation ---
        try:
            subs = pysubs2.load(extracted_sub_path, encoding="utf-8")
            if not subs: print(f"  Warning: Empty sub file. Skipping track.", file=sys.stderr); track_processing_failed = True; continue
            print(f"  Loaded {len(subs)} events from original ASS.")
        except Exception as e: print(f"  Error loading subs: {e}. Skipping track.", file=sys.stderr); track_processing_failed = True; continue

        # --- Translate ---
        if os.path.exists(translated_sub_path):
             print(f"  Cached translation: {os.path.basename(translated_sub_path)}")
             successfully_translated_files.append(translated_sub_path)
        else:
            print(f"  Translating {len(subs)} lines...")
            # Pass the client initialized within this worker AND the SRT context
            translated_subs = translate_subtitles_recursive(
                client=client,
                model_name=args.model, subs=subs, chunk_size=args.chunk_size,
                force_apply_mismatch=args.force_apply_mismatch,
                current_chunk_size=args.chunk_size,
                system_prompt=system_prompt, args=args,
                fallback_model_name=args.fallback_model,
                full_srt_context=full_srt_context # Pass the context here
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
        # If any track failed extraction/load/translate/save, count the whole file as failed.
        # Otherwise, if tracks existed but none were translated (e.g., all cached), it's not a failure.
        # If no tracks were found initially, it was returned as 'skipped' earlier.
        return 'failed' if track_processing_failed else 'skipped'

    print(f"\nMerging {len(successfully_translated_files)} track(s) for {filename}.")
    if not merge_subtitles(input_file_path, successfully_translated_files, output_file_path):
        print(f"Failed merge for {filename}."); return 'failed'

    print(f"Successfully processed: {filename}")
    return 'success'


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
    parser.add_argument("--fallback-model", help="Fallback model to try once on persistent single-line errors.")
    parser.add_argument("--api-delay", type=float, default=0.0, help="Optional delay in seconds between translation API calls for each chunk.")
    # Add parallel argument
    default_workers = os.cpu_count() if hasattr(os, 'cpu_count') else 4 # Default to 4 if cpu_count not available
    parser.add_argument("--parallel", nargs='?', type=int, const=default_workers, default=None,
                        help=f"Process N files in parallel. If N is omitted, defaults to {default_workers} (CPU cores or 4).")


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

    # --- List Models ---
    # Initialize client *only* if listing models, as workers/sequential loop will init their own
    if args.list_models:
        temp_client = None
        try:
            temp_client = genai.Client(api_key=api_key)
            print("\nAvailable Models (supporting 'generateContent'):"); count=0
            for model in temp_client.models.list():
                 # Assuming all listed models support generateContent for simplicity here
                 # A more robust check might involve inspecting model.supported_generation_methods
                 print(f"- {model.name}")
                 count += 1
            if count == 0: print("  No compatible models found.")
            else: print(f"\nFound {count} compatible models.")
        except Exception as e:
            print(f"Error listing models: {e}", file=sys.stderr)
        finally:
            # Clean up the temporary client if it was created
            # (genai client doesn't have an explicit close method as of typical usage)
            pass
        sys.exit(0) # Exit after listing models

    # --- Directory Setup ---
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

    # --- Process Files ---
    processed_files, skipped_files, failed_files = 0, 0, 0
    files_to_process_paths = []
    try:
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]
        files_to_process_paths = [os.path.join(input_dir, f) for f in all_files]
    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(files_to_process_paths)} potential files.")

    if args.parallel is not None:
        # --- Parallel Processing ---
        max_workers = args.parallel # Already an int or the default const value
        print(f"\n--- Starting Parallel Processing (Max Workers: {max_workers}) ---")

        # Use functools.partial to fix arguments for the worker function
        worker_func = partial(process_single_file,
                              args=args,
                              system_prompt=system_prompt,
                              cache_dir=cache_dir,
                              output_dir=output_dir)

        results = []
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use map to process files and get results in order (or use submit for more control)
                results = list(executor.map(worker_func, files_to_process_paths)) # Collect results
        except Exception as e:
            print(f"\n--- Error during parallel execution: {e} ---", file=sys.stderr)
            # Attempt to gracefully shutdown? ProcessPoolExecutor usually handles this.
            failed_files = len(files_to_process_paths) # Assume all failed if pool crashes badly

        # Tally results from parallel execution
        for status in results:
            if status == 'success':
                processed_files += 1
            elif status == 'skipped':
                skipped_files += 1
            elif status == 'failed':
                failed_files += 1
            else:
                 print(f"Warning: Unknown status '{status}' received from worker.", file=sys.stderr)
                 failed_files += 1 # Count unknowns as failures

    else:
        # --- Sequential Processing ---
        print("\n--- Starting Sequential Processing ---")
        for file_path in files_to_process_paths:
            status = process_single_file(file_path, args, system_prompt, cache_dir, output_dir)
            if status == 'success':
                processed_files += 1
            elif status == 'skipped':
                skipped_files += 1
            elif status == 'failed':
                failed_files += 1
            else:
                 print(f"Warning: Unknown status '{status}' received from process_single_file.", file=sys.stderr)
                 failed_files += 1 # Count unknowns as failures


    # --- Final Summary ---
    print("\n--- Processing Summary ---")
    print(f"Successfully processed files: {processed_files}")
    print(f"Skipped (output exists, no subs, etc.): {skipped_files}")
    print(f"Failed (error during processing/merge): {failed_files}")

    # print("\nCache cleanup logic placeholder") # Consider implementing actual cleanup

if __name__ == "__main__":
    main()