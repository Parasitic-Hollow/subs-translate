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
from openai import OpenAI # Added for OpenRouter
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
DEFAULT_GEMINI_MODEL = "models/gemini-1.5-flash-latest"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o" # Default for OpenRouter
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

# --- Gemini Translation Functions ---
def translate_chunk(client, model_name, chunk, system_prompt=None, retry_count=0, max_retries=3, timeout=DEFAULT_TIMEOUT, full_srt_context=None):
    """Translates a chunk of subtitle text using the Google Generative AI API, optionally guided by a system prompt and full SRT context."""
    max_delay = RETRY_DELAY * (2 ** max_retries)
    delay = min(RETRY_DELAY * (2 ** retry_count), max_delay)
    full_prompt = ""
    if system_prompt: full_prompt += f"System Prompt:\n{system_prompt}\n\n---\n"
    if full_srt_context:
        print("    Including full SRT context in prompt.")
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
    response = None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            print(f"    Sending chunk (length {len(chunk.splitlines())}) to Gemini model {model_name} (Attempt {retry_count + 1}). Timeout: {timeout}s")
            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    safety_settings=[
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    ]
                )
            )
        except TimeoutException:
             print(f"    Gemini API call timed out after {timeout} seconds.", file=sys.stderr)
             return "[API Call Timed Out]"
        finally:
            signal.alarm(0)

        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                translated_text = response.candidates[0].content.parts[0].text.strip()
                print(f"    Received translation chunk from Gemini.")
                return translated_text
            else:
                finish_reason = response.candidates[0].finish_reason if response and response.candidates else "UNKNOWN"
                safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else []
                print(f"    Warning (Gemini): Received empty or unexpected response. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}", file=sys.stderr)
                if finish_reason == genai_types.FinishReason.SAFETY:
                    return "[Blocked by Safety Filter]"
                else:
                    return None
        except (AttributeError, IndexError, TypeError) as resp_err:
             print(f"    Error parsing Gemini response: {resp_err}. Response: {response}", file=sys.stderr)
             return None

    except google_exceptions.GoogleAPIError as e:
        print(f"    Gemini API Error: {e}", file=sys.stderr)
        if isinstance(e, (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError, google_exceptions.ResourceExhausted)):
            if retry_count < max_retries:
                print(f"    Retrying Gemini API error in {delay} seconds...")
                time.sleep(delay)
                return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout, full_srt_context)
            else: print(f"    Max retries for Gemini API error. Skipping chunk.", file=sys.stderr); return None
        else: print(f"    Non-retriable Gemini API error. Skipping chunk.", file=sys.stderr); return None
    except Exception as e:
        print(f"    Unexpected error (Gemini): {type(e).__name__}: {e}", file=sys.stderr)
        if retry_count < max_retries:
             print(f"    Retrying unexpected Gemini error in {delay} seconds...")
             time.sleep(delay)
             return translate_chunk(client, model_name, chunk, system_prompt, retry_count + 1, max_retries, timeout, full_srt_context)
        else: print(f"    Max retries for unexpected Gemini error. Skipping chunk.", file=sys.stderr); return None

def translate_subtitles_recursive(client, model_name, subs, chunk_size, force_apply_mismatch, current_chunk_size, system_prompt=None, args=None, fallback_model_name=None, full_srt_context=None):
    """Translates subtitles (Gemini), chunking them, skipping complex tag lines, and handling retries."""
    initial_chunk_size = chunk_size
    print(f"  Translating subtitles using Gemini (model: {model_name}, initial chunk size: {initial_chunk_size})")

    original_lines = [event.text for event in subs]
    total_original_lines = len(original_lines)
    lines_to_translate_info = []
    skipped_line_count = 0
    for idx, line in enumerate(original_lines):
        is_complex_tag = re.search(r'\\t\(', line) or re.search(r'\\p', line)
        if not is_complex_tag:
            lines_to_translate_info.append((idx, line))
        else:
            print(f"  Skipping line {idx+1} (Gemini) because it contains complex tag: {line[:60]}...")
            skipped_line_count += 1

    translatable_texts = [text for _, text in lines_to_translate_info]
    total_translatable_lines = len(translatable_texts)
    print(f"  Identified {total_translatable_lines} lines for Gemini translation (skipped {skipped_line_count}).")

    if total_translatable_lines == 0:
        print("  No lines for Gemini translation. Returning original subtitles.")
        return subs

    final_translated_lines = list(original_lines)
    all_translated_results = []

    processed_translatable_idx = 0
    while processed_translatable_idx < total_translatable_lines:
        attempt_chunk_size = min(initial_chunk_size, total_translatable_lines - processed_translatable_idx)
        chunk_to_translate_list = translatable_texts[processed_translatable_idx : processed_translatable_idx + attempt_chunk_size]
        chunk_to_translate = "\n".join(chunk_to_translate_list)
        expected_chunk_len = len(chunk_to_translate_list)

        print(f"  Processing Gemini translatable chunk (idx {processed_translatable_idx}, attempting {expected_chunk_len} lines)")

        current_timeout = args.timeout if args else DEFAULT_TIMEOUT
        translated_chunk = translate_chunk(client, model_name, chunk_to_translate, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)

        retry_chunk_size = expected_chunk_len
        lines_added_this_iteration = None
        lines_processed_this_iteration = 0
        fallback_attempts = {}

        while lines_added_this_iteration is None:
            current_translatable_line_index = processed_translatable_idx

            if translated_chunk is None or translated_chunk == "[API Call Timed Out]":
                failure_reason = "timed out" if translated_chunk == "[API Call Timed Out]" else "failed"
                if retry_chunk_size <= 1:
                    if fallback_model_name:
                        fallback_count = fallback_attempts.get(current_translatable_line_index, 0)
                        if fallback_count < 3:
                            fallback_attempts[current_translatable_line_index] = fallback_count + 1
                            print(f"  Gemini translation {failure_reason} for single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). Fallback (try {fallback_count + 1}/3): {fallback_model_name}")
                            single_line_text_list = translatable_texts[current_translatable_line_index:min(current_translatable_line_index + 1, total_translatable_lines)]
                            if not single_line_text_list:
                                print(f"  Error: Gemini fallback slicing empty at index {current_translatable_line_index}. Skipping.", file=sys.stderr)
                                lines_added_this_iteration = ["[Fallback Slicing Error]"]
                                lines_processed_this_iteration = 1; break
                            single_line_text = "\n".join(single_line_text_list)
                            translated_chunk = translate_chunk(client, fallback_model_name, single_line_text, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                            continue
                        else:
                            print(f"  Failed Gemini single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}) after 3 fallbacks.", file=sys.stderr)
                            lines_added_this_iteration = ["[Translation Failed]"]; lines_processed_this_iteration = 1; break
                    else:
                        print(f"  Gemini translation {failure_reason} for single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). No fallback. Skipping.", file=sys.stderr)
                        lines_added_this_iteration = ["[Translation Failed]"]; lines_processed_this_iteration = 1; break
                else:
                    retry_chunk_size = max(1, retry_chunk_size - 5)
                    print(f"  Gemini translation {failure_reason} for chunk (idx {processed_translatable_idx}). Retrying with size {retry_chunk_size}.")
                    chunk_to_translate_retry_list = translatable_texts[processed_translatable_idx:min(processed_translatable_idx + retry_chunk_size, total_translatable_lines)]
                    if not chunk_to_translate_retry_list:
                         print(f"  Error: Gemini retry slicing empty at index {processed_translatable_idx}. Skipping original chunk.", file=sys.stderr)
                         lines_added_this_iteration = ["[Retry Slicing Error]" for _ in range(expected_chunk_len)]; lines_processed_this_iteration = expected_chunk_len; break
                    chunk_to_translate_retry = "\n".join(chunk_to_translate_retry_list)
                    translated_chunk = translate_chunk(client, model_name, chunk_to_translate_retry, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)

            elif translated_chunk == "[Blocked by Safety Filter]":
                print(f"  Gemini chunk (idx {processed_translatable_idx}) blocked by safety. Placeholder.", file=sys.stderr)
                lines_added_this_iteration = ["[Blocked by Safety Filter]" for _ in range(expected_chunk_len)]
                lines_processed_this_iteration = expected_chunk_len; break

            else: # Successful text
                chunk_translated_lines = translated_chunk.splitlines()
                last_sent_chunk_len = retry_chunk_size
                last_translated_chunk_len = len(chunk_translated_lines)

                if last_translated_chunk_len == last_sent_chunk_len:
                    print(f"  Successfully translated Gemini chunk (size {last_sent_chunk_len}, idx {processed_translatable_idx}).")
                    cleaned_lines = []
                    for i, line in enumerate(chunk_translated_lines):
                        if '<' in line and '>' in line:
                            original_line_preview = line[:60] + '...' if len(line) > 60 else line
                            stripped_line = re.sub(r"<[^>]+>", "", line).strip()
                            if stripped_line != line:
                                print(f"    Stripped tags (Gemini) line {i+1} (original index {lines_to_translate_info[processed_translatable_idx + i][0]+1}): '{original_line_preview}' -> '{stripped_line}'")
                            cleaned_lines.append(stripped_line)
                        else:
                            cleaned_lines.append(line)
                    lines_added_this_iteration = cleaned_lines
                    lines_processed_this_iteration = last_sent_chunk_len; break
                else: # Mismatch
                    print(f"    Warning (Gemini): Mismatch (size {last_sent_chunk_len}, idx {processed_translatable_idx}). Original: {last_sent_chunk_len}, Translated: {last_translated_chunk_len}", file=sys.stderr)
                    if force_apply_mismatch:
                        print("    Applying mismatched Gemini translation (--force-apply-mismatch).")
                        if last_translated_chunk_len < last_sent_chunk_len:
                            chunk_translated_lines.extend(["[Missing Translation]"] * (last_sent_chunk_len - last_translated_chunk_len))
                        else:
                            chunk_translated_lines = chunk_translated_lines[:last_sent_chunk_len]
                        cleaned_lines = []
                        for i, line in enumerate(chunk_translated_lines): # Strip tags if force applying
                            if '<' in line and '>' in line:
                                stripped_line = re.sub(r"<[^>]+>", "", line).strip()
                                cleaned_lines.append(stripped_line)
                            else:
                                cleaned_lines.append(line)
                        lines_added_this_iteration = cleaned_lines
                        lines_processed_this_iteration = last_sent_chunk_len; break
                    elif retry_chunk_size <= 1:
                        current_translatable_line_index = processed_translatable_idx
                        if fallback_model_name:
                            fallback_count = fallback_attempts.get(current_translatable_line_index, 0)
                            if fallback_count < 3:
                                fallback_attempts[current_translatable_line_index] = fallback_count + 1
                                print(f"    Gemini mismatch single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). Fallback (try {fallback_count + 1}/3): {fallback_model_name}")
                                single_line_text_list = translatable_texts[current_translatable_line_index:min(current_translatable_line_index + 1, total_translatable_lines)]
                                if not single_line_text_list:
                                    print(f"    Error: Gemini fallback (mismatch) slicing empty at index {current_translatable_line_index}. Skipping.", file=sys.stderr)
                                    lines_added_this_iteration = ["[Fallback Slicing Error (Mismatch)]"]; lines_processed_this_iteration = 1; break
                                single_line_text = "\n".join(single_line_text_list)
                                translated_chunk = translate_chunk(client, fallback_model_name, single_line_text, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)
                                continue
                            else:
                                print(f"    Failed Gemini single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}) mismatch after 3 fallbacks.", file=sys.stderr)
                                lines_added_this_iteration = ["[Translation Mismatch]"]; lines_processed_this_iteration = 1; break
                        else:
                            print(f"    Gemini mismatch single line (original index {lines_to_translate_info[current_translatable_line_index][0]+1}). No fallback. Skipping.", file=sys.stderr)
                            lines_added_this_iteration = ["[Translation Mismatch]"]; lines_processed_this_iteration = 1; break
                    else:
                        retry_chunk_size = max(1, retry_chunk_size // 2)
                        print(f"    Retrying Gemini with smaller chunk size: {retry_chunk_size}")
                        # Re-slice for the retry
                        chunk_to_translate_retry_list = translatable_texts[processed_translatable_idx:min(processed_translatable_idx + retry_chunk_size, total_translatable_lines)]
                        if not chunk_to_translate_retry_list: # Should not happen if retry_chunk_size >= 1
                             print(f"  Error: Gemini retry (mismatch) slicing resulted in empty chunk at index {processed_translatable_idx}. Skipping original chunk attempt.", file=sys.stderr)
                             lines_added_this_iteration = ["[Retry Slicing Error (Mismatch)]" for _ in range(expected_chunk_len)]; lines_processed_this_iteration = expected_chunk_len; break
                        chunk_to_translate_retry = "\n".join(chunk_to_translate_retry_list)
                        translated_chunk = translate_chunk(client, model_name, chunk_to_translate_retry, system_prompt=system_prompt, timeout=current_timeout, full_srt_context=full_srt_context)

            if lines_added_this_iteration is None: # Should not happen if logic is correct
                print(f"  Error: Gemini internal logic error (idx {processed_translatable_idx}). Skipping original chunk.", file=sys.stderr)
                lines_added_this_iteration = ["[Internal Logic Error]" for _ in range(expected_chunk_len)]
                lines_processed_this_iteration = expected_chunk_len

        if lines_added_this_iteration:
             all_translated_results.extend(lines_added_this_iteration)
             processed_translatable_idx += lines_processed_this_iteration
        else: # Should be caught above
             print(f"  Critical Error: Gemini loop exited without results for chunk at {processed_translatable_idx}. Advancing to prevent infinite loop.", file=sys.stderr)
             all_translated_results.extend(["[Critical Loop Error]" for _ in range(expected_chunk_len)])
             processed_translatable_idx += expected_chunk_len

        if args and args.api_delay > 0:
            print(f"  Applying API delay (Gemini): {args.api_delay}s")
            time.sleep(args.api_delay)

    if len(all_translated_results) != total_translatable_lines:
        print(f"  Warning (Gemini): Mismatch results ({len(all_translated_results)}) vs translatable ({total_translatable_lines}). Adjusting.", file=sys.stderr)
        if len(all_translated_results) < total_translatable_lines:
            all_translated_results.extend(["[Result Count Mismatch Error]"] * (total_translatable_lines - len(all_translated_results)))
        else:
            all_translated_results = all_translated_results[:total_translatable_lines]

    for i, (original_idx, _) in enumerate(lines_to_translate_info):
        if i < len(all_translated_results):
            final_translated_lines[original_idx] = all_translated_results[i]
        else:
            final_translated_lines[original_idx] = "[Missing Result Error]"


    if len(final_translated_lines) != total_original_lines:
         print(f"  FATAL Error (Gemini): Final line count ({len(final_translated_lines)}) != original ({total_original_lines}).", file=sys.stderr)
         return subs

    for idx, event in enumerate(subs):
        if idx < len(final_translated_lines):
             event.text = final_translated_lines[idx]
        else:
             print(f"  Error (Gemini): Mismatch applying final translation at index {idx}.", file=sys.stderr); break

    print(f"  Finished Gemini translation for {total_original_lines} original lines ({skipped_line_count} skipped).")
    return subs

# --- OpenRouter Translation Functions ---
def translate_chunk_openrouter(openrouter_client, model_name, chunk_text, system_prompt, full_srt_context, args, timeout):
    """Translates a chunk of subtitle text using the OpenRouter API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content_parts = []
    if full_srt_context:
        user_content_parts.append(f"Full Subtitle Context (SRT Format):\n{full_srt_context}\n---")

    user_content_parts.append(f"""IMPORTANT INSTRUCTION: Preserve any text enclosed in curly braces (e.g., {{{{\\an5\\i1}}}}) exactly as it appears in the original text and in its original position relative to the words. Do NOT translate the content within the curly braces.
CRITICAL: Pay close attention to context (including the full subtitle context provided above if present) to ensure accurate translation, especially regarding gender and number agreement.

Translate the following {SOURCE_LANG_NAME} subtitle text chunk to {TARGET_LANG_NAME}.
Preserve the original meaning, tone, and context.
Do not add any extra explanations, introductions, or formatting beyond the translated text itself.
Only output the translated text corresponding to the input lines.
Input Text Chunk:
---
{chunk_text}
---
Translated Text ({TARGET_LANG_NAME}):""")
    messages.append({"role": "user", "content": "\n".join(user_content_parts)})

    extra_headers = {}
    if args.openrouter_site_url:
        extra_headers["HTTP-Referer"] = args.openrouter_site_url
    if args.openrouter_site_name:
        extra_headers["X-Title"] = args.openrouter_site_name
    
    effective_extra_headers = extra_headers if extra_headers else None

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        print(f"    Sending chunk (length {len(chunk_text.splitlines())}) to OpenRouter model {model_name}. Timeout: {timeout}s")
        completion = openrouter_client.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_headers=effective_extra_headers
        )
        signal.alarm(0) 

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            translated_text = completion.choices[0].message.content.strip()
            print(f"    Received translation chunk from OpenRouter.")
            return translated_text
        else:
            print(f"    Warning (OpenRouter): Returned empty or unexpected response. Response: {completion}", file=sys.stderr)
            return None 
    except TimeoutException:
        signal.alarm(0) 
        print(f"    OpenRouter API call timed out after {timeout} seconds.", file=sys.stderr)
        return "[API Call Timed Out]"
    except Exception as e: 
        signal.alarm(0)
        print(f"    Error translating chunk with OpenRouter: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    finally:
        signal.alarm(0)

def translate_subtitles_recursive_openrouter(openrouter_client, model_name, subs, chunk_size, system_prompt, args, full_srt_context):
    """Translates subtitles using OpenRouter, with simple chunking and retries."""
    print(f"  Translating subtitles using OpenRouter (model: {model_name}, chunk size: {chunk_size})")
    original_lines = [event.text for event in subs]
    total_original_lines = len(original_lines)
    lines_to_translate_info = []
    skipped_line_count = 0

    for idx, line in enumerate(original_lines):
        is_complex_tag = re.search(r'\\t\(', line) or re.search(r'\\p', line)
        if not is_complex_tag:
            lines_to_translate_info.append((idx, line))
        else:
            print(f"  Skipping line {idx+1} (OpenRouter) due to complex tag: {line[:60]}...")
            skipped_line_count += 1

    translatable_texts = [text for _, text in lines_to_translate_info]
    total_translatable_lines = len(translatable_texts)
    print(f"  Identified {total_translatable_lines} lines for OpenRouter translation (skipped {skipped_line_count}).")

    if total_translatable_lines == 0:
        print("  No lines for OpenRouter translation. Returning original subtitles.")
        return subs

    final_translated_lines = list(original_lines) 
    all_translated_results = [] 

    processed_translatable_idx = 0
    max_retries_per_chunk = 3 

    while processed_translatable_idx < total_translatable_lines:
        current_chunk_list = translatable_texts[processed_translatable_idx : min(processed_translatable_idx + chunk_size, total_translatable_lines)]
        current_chunk_text = "\n".join(current_chunk_list)
        expected_lines_in_chunk = len(current_chunk_list)

        print(f"  Processing OpenRouter translatable chunk (idx {processed_translatable_idx}, {expected_lines_in_chunk} lines)")

        translated_chunk_text_result = None
        for attempt in range(max_retries_per_chunk):
            translated_chunk_text_result = translate_chunk_openrouter(
                openrouter_client, model_name, current_chunk_text,
                system_prompt, None, args, args.timeout
            )
            if translated_chunk_text_result and translated_chunk_text_result != "[API Call Timed Out]": # None is also a failure
                break 
            print(f"    Attempt {attempt + 1}/{max_retries_per_chunk} failed for OpenRouter chunk. Reason: {translated_chunk_text_result}")
            if attempt < max_retries_per_chunk - 1:
                time.sleep(RETRY_DELAY) 
        
        if translated_chunk_text_result and translated_chunk_text_result != "[API Call Timed Out]":
            translated_lines_for_chunk = translated_chunk_text_result.splitlines()
            cleaned_lines = []
            for i, line in enumerate(translated_lines_for_chunk):
                if '<' in line and '>' in line:
                    original_line_preview = line[:60] + '...' if len(line) > 60 else line
                    stripped_line = re.sub(r"<[^>]+>", "", line).strip()
                    if stripped_line != line:
                        print(f"    Stripped tags from OpenRouter line {i+1} (original index {lines_to_translate_info[processed_translatable_idx + i][0]+1}): '{original_line_preview}' -> '{stripped_line}'")
                    cleaned_lines.append(stripped_line)
                else:
                    cleaned_lines.append(line)
            translated_lines_for_chunk = cleaned_lines

            if len(translated_lines_for_chunk) == expected_lines_in_chunk:
                all_translated_results.extend(translated_lines_for_chunk)
            else: 
                print(f"    Warning (OpenRouter): Mismatch for chunk. Expected {expected_lines_in_chunk}, got {len(translated_lines_for_chunk)}.", file=sys.stderr)
                if args.force_apply_mismatch:
                    print("    Applying mismatched OpenRouter translation (--force-apply-mismatch).")
                    if len(translated_lines_for_chunk) < expected_lines_in_chunk:
                        translated_lines_for_chunk.extend(["[Missing Translation OR]"] * (expected_lines_in_chunk - len(translated_lines_for_chunk)))
                    else:
                        translated_lines_for_chunk = translated_lines_for_chunk[:expected_lines_in_chunk]
                    all_translated_results.extend(translated_lines_for_chunk)
                else:
                    all_translated_results.extend(["[Translation Mismatch OR]"] * expected_lines_in_chunk)
        else: 
            print(f"    Failed to translate OpenRouter chunk after {max_retries_per_chunk} attempts. Using placeholders.", file=sys.stderr)
            placeholder = "[API Call Timed Out OR]" if translated_chunk_text_result == "[API Call Timed Out]" else "[Translation Failed OR]"
            all_translated_results.extend([placeholder] * expected_lines_in_chunk)

        processed_translatable_idx += expected_lines_in_chunk
        if args.api_delay > 0: 
            print(f"  Applying API delay (OpenRouter): {args.api_delay}s")
            time.sleep(args.api_delay)

    if len(all_translated_results) != total_translatable_lines:
         print(f"  Warning (OpenRouter): Mismatch results ({len(all_translated_results)}) vs translatable ({total_translatable_lines}). Adjusting.", file=sys.stderr)
         if len(all_translated_results) < total_translatable_lines:
             all_translated_results.extend(["[OR Result Count Mismatch Error]"] * (total_translatable_lines - len(all_translated_results)))
         else:
             all_translated_results = all_translated_results[:total_translatable_lines]

    for i, (original_idx, _) in enumerate(lines_to_translate_info):
        if i < len(all_translated_results):
            final_translated_lines[original_idx] = all_translated_results[i]
        else: 
            final_translated_lines[original_idx] = "[OR Missing Result Error]"

    for idx, event in enumerate(subs):
        if idx < len(final_translated_lines):
            event.text = final_translated_lines[idx]
        else:
            print(f"  Error (OpenRouter): Mismatch applying final translation at index {idx}.", file=sys.stderr); break

    print(f"  Finished OpenRouter translation for {total_original_lines} original lines ({skipped_line_count} skipped).")
    return subs

# --- MKVMerge Function ---
def merge_subtitles(input_video_file, translated_subtitle_files, output_video_file):
    """Merges multiple translated subtitle files into the video using mkvmerge."""
    if not translated_subtitle_files:
        print("  No translated subtitles provided for merging. Skipping merge step.", file=sys.stderr)
        return False

    print(f"Merging {len(translated_subtitle_files)} translated subtitle track(s) into: {os.path.basename(output_video_file)}")
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    command = [ "mkvmerge", "-o", output_video_file, input_video_file ] 

    for sub_file in translated_subtitle_files:
        track_name_suffix = ""
        match = re.search(r'\.track_(\d+)\.', sub_file)
        if match: track_name_suffix = f" (Track {match.group(1)})"
        command.extend([
            "--language", f"0:{TARGET_LANG_CODE}",
            "--track-name", f"0:{TARGET_LANG_NAME}", 
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
    translation_client = None 
    model_to_use = None
    
    if args.use_openrouter:
        print(f"  Using OpenRouter for translation.")
        openrouter_api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print(f"Error: OpenRouter API Key not found. Set --openrouter-api-key or OPENROUTER_API_KEY env var. Skipping file.", file=sys.stderr)
            return 'failed'
        try:
            translation_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            model_to_use = args.openrouter_model
            print(f"  OpenRouter Client initialized. Model: {model_to_use}")
        except Exception as e:
            print(f"Error initializing OpenRouter Client: {e}. Skipping file.", file=sys.stderr)
            return 'failed'
    else: 
        print(f"  Using Gemini for translation.")
        google_api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print(f"Error: Google API Key not found. Set --api-key or GOOGLE_API_KEY env var. Skipping file.", file=sys.stderr)
            return 'failed'
        try:
            translation_client = genai.Client(api_key=google_api_key)
            model_to_use = args.model 
            print(f"  Google AI Client initialized. Model: {model_to_use}")
        except Exception as e:
            print(f"Error initializing Google AI Client: {e}. Skipping file.", file=sys.stderr)
            return 'failed'

    all_subtitle_tracks = get_subtitle_tracks(input_file_path)
    if not all_subtitle_tracks: print("No subs found. Skipping."); return 'skipped'

    english_tracks = { idx: info for idx, info in all_subtitle_tracks.items() if info['lang'] == SOURCE_LANG_CODE }
    if not english_tracks: print(f"No {SOURCE_LANG_NAME} tracks. Skipping."); return 'skipped'

    print(f"Found {len(english_tracks)} {SOURCE_LANG_NAME} track(s): {list(english_tracks.keys())}")
    successfully_translated_files = []
    track_processing_failed = False

    for track_index, track_info in english_tracks.items():
        print(f"\n  -- Track Index: {track_index} --")
        extracted_sub_path = os.path.join(cache_dir, f"{sanitized_base_name}.{SOURCE_LANG_CODE}.track_{track_index}.ass")
        translated_sub_path = os.path.join(cache_dir, f"{sanitized_base_name}.{TARGET_LANG_CODE}.track_{track_index}.ass")

        if os.path.exists(extracted_sub_path): print(f"  Cached source: {os.path.basename(extracted_sub_path)}")
        elif not extract_subtitle(input_file_path, track_index, extracted_sub_path):
            print(f"  Failed extract track {track_index}. Skipping track."); track_processing_failed = True; continue

        temp_srt_path = extracted_sub_path + ".temp.srt"
        full_srt_context = None
        if not args.use_openrouter and args.no_srt_context:
            print("  Skipping SRT context generation for Gemini as per --no-srt-context.")
            full_srt_context = None
        elif not args.use_openrouter: # This is the Gemini path, and --no-srt-context is false
            try:
                print(f"  Converting extracted ASS to SRT for context: {os.path.basename(temp_srt_path)}")
                convert_command = ["ffmpeg", "-y", "-i", extracted_sub_path, temp_srt_path]
                subprocess.run(convert_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
                if os.path.exists(temp_srt_path) and os.path.getsize(temp_srt_path) > 0:
                    with open(temp_srt_path, 'r', encoding='utf-8') as f_srt: full_srt_context = f_srt.read()
                    print(f"  Read {len(full_srt_context.splitlines())} lines from temporary SRT for context.")
                else: print(f"  Warning: SRT conversion output missing or empty.", file=sys.stderr)
            except Exception as e: print(f"  Error/Warning during SRT conversion: {e}", file=sys.stderr)
            finally:
                if os.path.exists(temp_srt_path):
                    try: os.remove(temp_srt_path)
                    except OSError as rm_err: print(f"  Warning: Failed to delete temp SRT: {rm_err}", file=sys.stderr)
        # For OpenRouter, full_srt_context is not used by translate_subtitles_recursive_openrouter,
        # but we'll keep the SRT generation for potential future use or consistency if desired.
        # If OpenRouter needs to skip SRT, a similar flag would be needed for it.
        # Currently, translate_chunk_openrouter receives full_srt_context=None directly.
        # The existing logic for OpenRouter in translate_subtitles_recursive_openrouter passes None for full_srt_context.
        # So, if args.use_openrouter is true, the SRT context is generated here but not used by the OpenRouter translation function.
        # This is fine as per current requirements.
        
        try:
            subs = pysubs2.load(extracted_sub_path, encoding="utf-8")
            if not subs: print(f"  Warning: Empty sub file. Skipping track.", file=sys.stderr); track_processing_failed = True; continue
            print(f"  Loaded {len(subs)} events from original ASS.")
        except Exception as e: print(f"  Error loading subs: {e}. Skipping track.", file=sys.stderr); track_processing_failed = True; continue

        if os.path.exists(translated_sub_path):
             print(f"  Cached translation: {os.path.basename(translated_sub_path)}")
             successfully_translated_files.append(translated_sub_path)
        else:
            print(f"  Translating {len(subs)} lines...")
            translated_subs = None
            if args.use_openrouter:
                translated_subs = translate_subtitles_recursive_openrouter(
                    openrouter_client=translation_client, 
                    model_name=model_to_use, 
                    subs=subs,
                    chunk_size=args.chunk_size,
                    system_prompt=system_prompt,
                    args=args, 
                    full_srt_context=full_srt_context
                )
            else: 
                translated_subs = translate_subtitles_recursive(
                    client=translation_client, 
                    model_name=model_to_use, 
                    subs=subs,
                    chunk_size=args.chunk_size,
                    force_apply_mismatch=args.force_apply_mismatch,
                    current_chunk_size=args.chunk_size, 
                    system_prompt=system_prompt,
                    args=args, 
                    fallback_model_name=args.fallback_model,
                    full_srt_context=full_srt_context
                )
            
            if translated_subs is None: print("  Translation failed/timed out. Skipping track."); track_processing_failed = True; continue
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

    if not successfully_translated_files:
        print(f"\nNo tracks translated for {filename}. Skipping merge.")
        return 'failed' if track_processing_failed else 'skipped'

    print(f"\nMerging {len(successfully_translated_files)} track(s) for {filename}.")
    if not merge_subtitles(input_file_path, successfully_translated_files, output_file_path):
        print(f"Failed merge for {filename}."); return 'failed'

    print(f"Successfully processed: {filename}")
    return 'success'


# --- Main Function ---
def main():
    """Main function to parse arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(description=f"Extracts ALL {SOURCE_LANG_NAME} subtitles, translates to {TARGET_LANG_NAME} (Gemini/OpenRouter), and merges.")

    # Gemini specific
    parser.add_argument("--api-key", help="Google Generative AI API Key (for Gemini). Can use GOOGLE_API_KEY env var.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help=f"Gemini model for translation (default: {DEFAULT_GEMINI_MODEL}).")
    parser.add_argument("--fallback-model", help="Gemini fallback model for persistent single-line errors.")
    parser.add_argument("--list-models", action="store_true", help="List compatible Gemini models and exit.")
    parser.add_argument("--no-srt-context", action="store_true", help="Do not send the full SRT content as context to the Gemini API.")

    # OpenRouter specific
    parser.add_argument("--use-openrouter", action='store_true', help="Use OpenRouter for translation instead of Gemini.")
    parser.add_argument("--openrouter-api-key", help="OpenRouter API Key. Can use OPENROUTER_API_KEY env var.")
    parser.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL, help=f"OpenRouter model name (default: {DEFAULT_OPENROUTER_MODEL}).")
    parser.add_argument("--openrouter-site-url", help="Your site URL for OpenRouter (optional HTTP-Referer header).")
    parser.add_argument("--openrouter-site-name", help="Your site name for OpenRouter (optional X-Title header).")

    # Common arguments
    parser.add_argument("--directory", default=".", help="Directory containing video files.")
    parser.add_argument("--output-dir", default="merged_output", help="Directory for merged videos.")
    parser.add_argument("--cache-dir", default="subtrans_cache", help="Directory for temp subs.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before starting.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Lines per API request (default: {DEFAULT_CHUNK_SIZE}).")
    parser.add_argument("--force-apply-mismatch", action="store_true", help="Apply translation even if line counts mismatch (both APIs).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout per translation API call (default: {DEFAULT_TIMEOUT}s).")
    parser.add_argument("--system-prompt-file", help="Path to system prompt text file (used by both APIs).")
    parser.add_argument("--api-delay", type=float, default=0.0, help="Optional delay (seconds) between translation API calls for each chunk.")
    
    default_workers = os.cpu_count() if hasattr(os, 'cpu_count') else 4
    parser.add_argument("--parallel", nargs='?', type=int, const=default_workers, default=None,
                        help=f"Process N files in parallel. If N omitted, defaults to {default_workers} (CPU cores or 4).")

    args = parser.parse_args()

    system_prompt = None
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, 'r', encoding='utf-8') as f: system_prompt = f.read().strip()
            if system_prompt: print(f"Using system prompt: {args.system_prompt_file}")
            else: print(f"Warning: System prompt file empty.", file=sys.stderr)
        except Exception as e: print(f"Error reading system prompt file: {e}", file=sys.stderr)

    print("--- Subtitle Extraction and Translation Script ---")
    print(f"Target Language: {TARGET_LANG_NAME} ({TARGET_LANG_CODE})")
    if args.use_openrouter:
        print(f"Translation API: OpenRouter (Model: {args.openrouter_model})")
    else:
        print(f"Translation API: Gemini (Model: {args.model})")


    if not shutil.which("ffmpeg"): print("Error: ffmpeg not found.", file=sys.stderr); sys.exit(1)
    if not shutil.which("mkvmerge"): print("Error: mkvmerge not found.", file=sys.stderr); sys.exit(1)
    print("Dependencies (ffmpeg, mkvmerge) found.")

    # API Key check for Gemini if listing models or not using OpenRouter
    if args.list_models or not args.use_openrouter:
        gemini_api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key: 
            print("Error: Google API Key not found (required for Gemini or --list-models). Set --api-key or GOOGLE_API_KEY env var.", file=sys.stderr)
            if not args.use_openrouter or args.list_models : sys.exit(1) # Exit if Gemini is primary or listing
        else:
            print("Google API Key configured (for Gemini / model listing).")
    
    # API Key check for OpenRouter if selected
    if args.use_openrouter:
        openrouter_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            print("Error: OpenRouter API Key not found. Set --openrouter-api-key or OPENROUTER_API_KEY env var.", file=sys.stderr)
            sys.exit(1)
        else:
            print("OpenRouter API Key configured.")


    if args.list_models: # This lists Gemini models
        gemini_api_key_for_list = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key_for_list:
             print("Error: Google API Key needed to list models.", file=sys.stderr); sys.exit(1)
        try:
            temp_client = genai.Client(api_key=gemini_api_key_for_list)
            print("\nAvailable Gemini Models (supporting 'generateContent'):"); count=0
            for model_info in temp_client.models.list():
                 print(f"- {model_info.name}")
                 count += 1
            if count == 0: print("  No compatible Gemini models found.")
            else: print(f"\nFound {count} compatible Gemini models.")
        except Exception as e: print(f"Error listing Gemini models: {e}", file=sys.stderr)
        sys.exit(0)

    script_dir = pathlib.Path(__file__).parent.resolve()
    input_dir = os.path.abspath(args.directory)
    output_dir = script_dir / args.output_dir
    cache_dir = os.path.abspath(args.cache_dir)
    print(f"Input: {input_dir}\nOutput: {output_dir}\nCache: {cache_dir}")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)

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

    processed_files, skipped_files, failed_files = 0, 0, 0
    files_to_process_paths = []
    try:
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]
        files_to_process_paths = [os.path.join(input_dir, f) for f in all_files]
    except FileNotFoundError: print(f"Error: Input directory not found: {input_dir}", file=sys.stderr); sys.exit(1)
    print(f"\nFound {len(files_to_process_paths)} potential files.")

    if args.parallel is not None:
        max_workers = args.parallel 
        print(f"\n--- Starting Parallel Processing (Max Workers: {max_workers}) ---")
        worker_func = partial(process_single_file, args=args, system_prompt=system_prompt, cache_dir=cache_dir, output_dir=output_dir)
        results = []
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(worker_func, files_to_process_paths)) 
        except Exception as e:
            print(f"\n--- Error during parallel execution: {e} ---", file=sys.stderr)
            failed_files = len(files_to_process_paths) 
        for status in results:
            if status == 'success': processed_files += 1
            elif status == 'skipped': skipped_files += 1
            elif status == 'failed': failed_files += 1
            else: print(f"Warning: Unknown status '{status}' from worker.", file=sys.stderr); failed_files += 1
    else:
        print("\n--- Starting Sequential Processing ---")
        for file_path in files_to_process_paths:
            status = process_single_file(file_path, args, system_prompt, cache_dir, output_dir)
            if status == 'success': processed_files += 1
            elif status == 'skipped': skipped_files += 1
            elif status == 'failed': failed_files += 1
            else: print(f"Warning: Unknown status '{status}'.", file=sys.stderr); failed_files += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed files: {processed_files}")
    print(f"Skipped (output exists, no subs, etc.): {skipped_files}")
    print(f"Failed (error during processing/merge): {failed_files}")

if __name__ == "__main__":
    main()