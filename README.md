# MKV Subtitle Extractor & Translator

## Purpose

This Python script automates the process of extracting English Advanced SubStation Alpha (ASS) subtitles from video files. It then utilizes the Google Generative AI API to translate these subtitles into Latin American Spanish (es-419). Finally, it merges the translated subtitles into new video files in a specified output directory.

## Features

*   Checks for required external dependencies (`ffmpeg`, `mkvmerge`) before starting.
*   Processes video files found within a specified input directory.
*   Extracts English ASS subtitle tracks from video files using `ffmpeg`.
*   Translates the extracted subtitle text content to Latin American Spanish using the Google Generative AI API.
*   Handles potential translation errors and API issues with retries.
*   Caches extracted and translated subtitle files (`.ass`) to avoid redundant processing.
*   Merges the translated subtitles into new video files using `mkvmerge`, placing them in a designated output directory.
*   Provides command-line options for customization (API key, directories, model selection, etc.).
*   Skips processing videos if a corresponding merged file already exists in the output directory.

## Dependencies

### External Tools
The following command-line tools must be installed and accessible in your system's PATH:
*   **FFmpeg:** Used for extracting subtitle streams from video files. Download from [https://ffmpeg.org/](https://ffmpeg.org/).
*   **MKVToolNix:** The `mkvmerge` tool (part of MKVToolNix) is used for merging the translated subtitles back into new video files. Download from [https://mkvtoolnix.download/](https://mkvtoolnix.download/).

### Python Requirements
*   **Python 3.x:** The script is written for Python 3.
*   **Required Packages:** The necessary Python packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    *   `google-generativeai`: For interacting with the Google AI API.
    *   `pysubs2`: For parsing and manipulating subtitle files.
    *   (Other standard libraries are also used)

### Google API Key
*   You need a valid Google API Key with the Generative Language API (Gemini) enabled. Obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.

## Setup

1.  **Install External Tools:** Ensure `ffmpeg` and `mkvmerge` are installed and in your PATH.
2.  **Clone/Download:** Get the script and project files (including `requirements.txt`).
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .\.venv\Scripts\activate.bat # Windows CMD
    # .\.venv\Scripts\Activate.ps1 # Windows PowerShell
    ```
4.  **Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure API Key:**
    *   Set the `GOOGLE_API_KEY` environment variable (recommended):
        ```bash
        export GOOGLE_API_KEY='YOUR_API_KEY' # Linux/macOS
        # set GOOGLE_API_KEY=YOUR_API_KEY # Windows CMD
        # $env:GOOGLE_API_KEY='YOUR_API_KEY' # Windows PowerShell
        ```
    *   Or, use the `--api-key` command-line argument.

## Usage

Run the script from your terminal, ensuring your virtual environment is activated.

```bash
python extract_translate_subs.py [options]
```

**Command-Line Arguments:**

*   `--api-key YOUR_API_KEY`: Specifies the Google API Key directly. Overrides the `GOOGLE_API_KEY` environment variable. **(Required if environment variable is not set)**
*   `--directory PATH/TO/VIDEOS`: Directory containing the source video files to process. Defaults to the current directory (`.`).
*   `--output-dir PATH/TO/OUTPUT`: Directory where the merged video files will be saved. Defaults to `./merged_output`.
*   `--cache-dir PATH/TO/CACHE`: Directory to store intermediate/cached subtitle files (`.ass`). Defaults to `./subtrans_cache`.
*   `--clear-cache`: If specified, clears the cache directory before starting.
*   `--list-models`: Lists compatible Google AI models available via your API key and exits.
*   `--model MODEL_NAME`: Specifies the Google AI model to use for translation (e.g., `gemini-1.5-flash-latest`, `gemini-pro`). Defaults to `gemini-1.5-flash-latest`. Use `--list-models` to see options.
*   `--chunk-size N`: Number of subtitle lines to translate per API call (default: 15).
*   `--force-apply-mismatch`: Force apply translation even if line counts mismatch after retries. **Warning:** This can lead to desynchronized or incorrect subtitles. Use with caution.
*   `--timeout SECONDS`: Timeout in seconds for individual API translation calls (default: 180.0).
*   `--system-prompt-file FILEPATH`: Path to a text file containing a custom system prompt for the translation model.

**Example:**

Translate subtitles for videos in `/media/movies`, saving results to `/media/movies_translated`, using your API key:

```bash
export GOOGLE_API_KEY='YOUR_API_KEY' # Or use --api-key argument
python extract_translate_subs.py --directory /media/movies --output-dir /media/movies_translated
```

## Output

*   New video files with the translated subtitles merged in are created in the specified `--output-dir` (default: `./merged_output`). Original filenames are preserved.
*   The original video files in the source `--directory` remain untouched.
*   Intermediate subtitle files (`.eng.ass`, `.es-419.ass`) are stored in the `--cache-dir` (default: `./subtrans_cache`). These are used to skip re-extraction or re-translation on subsequent runs unless `--clear-cache` is used.

## Notes & Troubleshooting

*   **API Costs:** Using the Google Generative AI API may incur costs. Monitor your usage.
*   **Rate Limits:** The API has rate limits. Processing many files might trigger temporary errors.
*   **Tool Not Found:** Ensure `ffmpeg` and `mkvmerge` are installed and in your system's PATH.
*   **Translation Quality:** Depends on the model and source text complexity.
*   **Error Handling:** The script retries failed translation chunks with smaller sizes. Use `--force-apply-mismatch` cautiously for lines that fail repeatedly.
*   **No English ASS Track:** Videos without detectable English ASS subtitles will be skipped.
*   **Existing Output:** Files already present in the output directory will be skipped.