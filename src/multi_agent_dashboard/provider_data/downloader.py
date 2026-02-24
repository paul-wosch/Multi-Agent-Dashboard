"""
Download provider models from external source.

Fetches raw JSON from models.dev and saves it locally.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from multi_agent_dashboard.config import (
    DATA_PATH,
    PROVIDER_DATA_DIR,
    PROVIDER_MODELS_ALL_FILE,
    MODELS_DEV_URL,
)

logger = logging.getLogger(__name__)


def download_provider_models_all(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Download raw provider models JSON from models.dev.

    Implements exponential backoff retry logic for network errors.
    Saves the raw JSON to PROVIDER_MODELS_ALL_FILE.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        initial_backoff: Initial backoff time in seconds (default 1.0).
        timeout: Request timeout in seconds (default 30.0).

    Returns:
        Parsed JSON dictionary.

    Raises:
        ImportError: If the `requests` library is not installed.
        ConnectionError: If all retries fail or a non‑HTTP‑error occurs.
        ValueError: If the response is not valid JSON.
    """
    # Ensure the requests library is available
    try:
        import requests
    except ImportError:
        raise ImportError(
            "download_provider_models_all requires the 'requests' library. "
            "Install with: pip install requests"
        )

    # Compute target directory and file path
    provider_data_dir = DATA_PATH.parent / PROVIDER_DATA_DIR
    provider_data_dir.mkdir(parents=True, exist_ok=True)
    target_file = provider_data_dir / PROVIDER_MODELS_ALL_FILE

    # Prepare request headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Downloading provider models from {MODELS_DEV_URL} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            response = requests.get(
                MODELS_DEV_URL,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            # Parse JSON to validate
            data = response.json()

            # Save raw JSON
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Downloaded provider models, saved to {target_file} "
                f"({len(response.text)} bytes)"
            )
            return data

        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.warning(
                f"Download attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)  # exponential
                logger.debug(f"Waiting {backoff:.1f}s before retry")
                time.sleep(backoff)
            # else continue to raise after loop

    # All retries exhausted
    logger.error(
        f"Failed to download provider models after {max_retries} attempts"
    )
    raise ConnectionError(
        f"Could not download provider models from {MODELS_DEV_URL}"
    ) from last_exception