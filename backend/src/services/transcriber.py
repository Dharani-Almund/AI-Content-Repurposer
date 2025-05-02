import os
import logging
import traceback
import mimetypes
import importlib
from typing import Dict, Optional, Union, Tuple

import requests
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('podcast_transcriber.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class PodcastTranscriber:
    """
    Advanced podcast transcription utility with enhanced processing.
    """

    def __init__(self, model_size: str = 'base'):
        """
        Initialize transcription model.

        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        models = {
            'tiny': 'openai/whisper-tiny',
            'base': 'openai/whisper-base',
            'small': 'openai/whisper-small',
            'medium': 'openai/whisper-medium',
            'large': 'openai/whisper-large'
        }

        model_path = models.get(model_size, 'openai/whisper-base')

        # Ensure cache and temp directories
        os.makedirs('podcast_cache', exist_ok=True)
        os.makedirs('podcast_transcripts', exist_ok=True)

        # Load models
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)

    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if all required dependencies are available.

        Returns:
            Dict[str, bool]: Dictionary of dependency status
        """
        dependencies = {
            "requests": self._check_module("requests"),
            "librosa": self._check_module("librosa"),
            "transformers": self._check_module("transformers"),
            "torch": self._check_module("torch"),
            "audio_processing": self._check_audio_processing()
        }

        logger.info(f"Dependency check complete: {dependencies}")
        return dependencies

    def _check_module(self, module_name: str) -> bool:
        """
        Check if a Python module is available.

        Args:
            module_name (str): Name of the module to check

        Returns:
            bool: True if module is available, False otherwise
        """
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            logger.warning(f"Required module not found: {module_name}")
            return False

    def _check_audio_processing(self) -> bool:
        """
        Check if audio processing capabilities are available.

        Returns:
            bool: True if audio processing is available, False otherwise
        """
        try:
            # Create a small test audio file
            test_file = os.path.join('podcast_cache', 'test_audio.wav')
            if not os.path.exists(test_file):
                import numpy as np
                from scipy.io import wavfile
                sample_rate = 16000
                # Generate 1 second of silence
                data = np.zeros(sample_rate, dtype=np.int16)
                wavfile.write(test_file, sample_rate, data)

            # Try to load it with librosa
            _, sr = librosa.load(test_file, sr=16000, duration=0.1)
            return True
        except Exception as e:
            logger.warning(f"Audio processing check failed: {e}")
            return False

    def download_audio(
            self,
            audio_url: str,
            file_name: Optional[str] = None,
            max_size_mb: int = 200
    ) -> Optional[str]:
        """
        Advanced audio download with file type validation and size limits.

        Args:
            audio_url (str): URL of audio file
            file_name (Optional[str]): Custom filename
            max_size_mb (int): Maximum file size in MB

        Returns:
            Optional path to downloaded file
        """
        try:
            # Validate URL format
            if not audio_url or not isinstance(audio_url, str):
                logger.error(f"Invalid audio URL: {audio_url}")
                return None

            # Ensure URL has scheme
            if not audio_url.startswith(('http://', 'https://')):
                logger.error(f"URL missing scheme: {audio_url}")
                return None

            file_name = file_name or os.path.basename(audio_url.split("?")[0]) or "podcast_audio.mp3"
            file_path = os.path.join('podcast_cache', file_name)

            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'audio/*'
            }

            with requests.get(audio_url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Check file size
                content_length = int(response.headers.get('content-length', 0))
                if content_length > max_size_mb * 1024 * 1024:
                    logger.warning(f"File exceeds {max_size_mb}MB limit")
                    return None

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Validate file type using mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not mime_type.startswith('audio/'):
                os.remove(file_path)
                logger.warning(f"Invalid file type: {mime_type}")
                return None

            return file_path

        except Exception as e:
            logger.error(f"Audio download error: {e}")
            return None

    def transcribe_audio(
            self,
            audio_path: str,
            language: str = 'en',
            task: str = 'transcribe'
    ) -> Optional[Dict[str, Union[str, float]]]:
        """
        Advanced audio transcription with language support and handling for longer audio.

        Args:
            audio_path (str): Path to audio file
            language (str): Language code
            task (str): Transcription task ('transcribe' or 'translate')

        Returns:
            Transcription result with metadata
        """
        try:
            # Validate audio path
            if not audio_path or not isinstance(audio_path, str):
                logger.error(f"Invalid audio path: {audio_path}")
                return None

            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None

            # Load audio file
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = librosa.get_duration(y=waveform, sr=sr)
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")

            # Handle longer content by chunk processing
            if audio_duration > 30:  # If longer than 30 seconds
                # Process in chunks for longer audio
                chunk_duration = 30  # seconds per chunk
                chunk_samples = int(chunk_duration * sr)
                num_chunks = int(len(waveform) / chunk_samples) + 1

                logger.info(f"Processing audio in {num_chunks} chunks")

                full_transcription = ""

                for i in range(num_chunks):
                    chunk_start = i * chunk_samples
                    chunk_end = min(chunk_start + chunk_samples, len(waveform))
                    audio_chunk = waveform[chunk_start:chunk_end]

                    if len(audio_chunk) < sr:  # Skip chunks shorter than 1 second
                        continue

                    # Process this chunk
                    input_features = self.processor(
                        audio_chunk,
                        sampling_rate=sr,
                        return_tensors="pt"
                    ).input_features

                    # Make sure input features have valid content
                    if input_features.size(1) == 0:
                        logger.warning(f"Empty input features for chunk {i + 1}, skipping")
                        continue

                    # Get decoder prompt IDs
                    decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language,
                        task=task
                    )

                    try:
                        # Generate transcription for this chunk
                        predicted_ids = self.whisper_model.generate(
                            input_features,
                            max_length=448,  # Using a more stable value
                            num_beams=5,
                            forced_decoder_ids=decoder_ids
                        )

                        # Decode transcription
                        chunk_transcription = self.processor.batch_decode(
                            predicted_ids,
                            skip_special_tokens=True
                        )[0]

                        full_transcription += chunk_transcription + " "
                        logger.info(f"Chunk {i + 1}/{num_chunks} processed, current length: {len(full_transcription)}")

                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {i + 1}: {chunk_error}")
                        # Continue with next chunk instead of failing the whole process
                        continue

                transcription = full_transcription.strip()
            else:
                # Original process for shorter audio
                input_features = self.processor(
                    waveform,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features

                # Get decoder prompt IDs
                decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task=task
                )

                # Generate transcription
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    max_length=448,  # Using standard value
                    num_beams=5,
                    forced_decoder_ids=decoder_ids
                )

                # Decode transcription
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

            # Log transcript length
            logger.info(f"Transcription completed. Length: {len(transcription)} characters")

            # Generate transcript file
            os.makedirs('podcast_transcripts', exist_ok=True)  # Ensure directory exists
            transcript_path = os.path.join(
                'podcast_transcripts',
                os.path.splitext(os.path.basename(audio_path))[0] + '.txt'
            )

            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            logger.info(f"Transcript saved to {transcript_path}")

            return {
                'text': transcription,
                'language': language,
                'task': task,
                'transcript_path': transcript_path,
                'audio_duration': audio_duration
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return None