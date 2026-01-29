import base64
import os
import tempfile

import speech_recognition as sr
from pydub import AudioSegment

from mirix.log import get_logger

logger = get_logger(__name__)


def convert_base64_to_audio_segment(voice_file_b64):
    """Convert base64 voice data to AudioSegment using temporary file"""
    try:
        # Convert base64 to AudioSegment using temporary file
        audio_data = base64.b64decode(voice_file_b64)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Load AudioSegment from temporary file
        audio_segment = AudioSegment.from_file(temp_file_path)

        # Clean up temporary file
        os.unlink(temp_file_path)

        return audio_segment
    except Exception as e:
        logger.error("Error converting voice data to AudioSegment: %s", str(e))
        return None


def process_voice_files(voice_items):
    """Process accumulated voice files by concatenating them and return combined transcription"""
    if not voice_items:
        return None

    logger.debug("🎵 Agent processing %s voice files", len(voice_items))
    temp_files = []

    try:
        # Separate already-converted AudioSegments from base64 fallbacks
        audio_segments = []
        base64_items = []

        for i, (timestamp, item) in enumerate(voice_items):
            if item.get("is_base64", False):
                # Handle fallback base64 items
                base64_items.append((timestamp, item))
            else:
                # Handle already-converted AudioSegments
                audio_segments.append(item["content"])

        # Now concatenate all audio segments and transcribe
        if audio_segments:
            try:
                # Concatenate all audio segments
                combined_audio = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined_audio += segment

                # Create temporary file for combined audio
                temp_audio_file = None
                try:
                    # Create temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_audio_file = temp_file.name

                    # Export combined audio to temporary file
                    combined_audio.export(temp_audio_file, format="wav")

                    # Initialize speech recognizer
                    recognizer = sr.Recognizer()

                    # Perform speech recognition on the combined audio from temporary file
                    with sr.AudioFile(temp_audio_file) as source:
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source)
                        # Record the audio
                        audio_data = recognizer.record(source)

                        # Try to recognize speech using Google Speech Recognition
                        try:
                            transcription = recognizer.recognize_google(audio_data)
                            first_timestamp = voice_items[0][0] if voice_items else "unknown"
                            combined_transcription = f"[{first_timestamp}] {transcription}"
                            return combined_transcription

                        except sr.UnknownValueError:
                            logger.error("Could not understand combined audio")
                            return None

                        except sr.RequestError as e:
                            logger.error("Google Speech Recognition failed for combined audio: %s", str(e))
                            # Fallback to offline methods if Google fails
                            try:
                                transcription = recognizer.recognize_sphinx(audio_data)
                                first_timestamp = voice_items[0][0] if voice_items else "unknown"
                                combined_transcription = f"[{first_timestamp}] {transcription}"
                                logger.error(
                                    "Sphinx transcribed combined audio: '%s'",
                                    transcription[:100] + ("..." if len(transcription) > 100 else ""),
                                )
                                return combined_transcription
                            except Exception:
                                logger.error("All recognition methods failed for combined audio")
                                return None

                finally:
                    # Clean up temporary file
                    if temp_audio_file and os.path.exists(temp_audio_file):
                        try:
                            os.unlink(temp_audio_file)
                            logger.debug("Deleted temporary audio file: %s", temp_audio_file)
                        except Exception as cleanup_error:
                            logger.error(
                                "Failed to delete temporary audio file %s: %s", temp_audio_file, str(cleanup_error)
                            )

            except Exception as e:
                logger.error("Error in concatenation and transcription: %s", str(e))
                return None
        else:
            logger.debug("No valid audio segments to process")
            return None

    except Exception as e:
        logger.exception("Critical error in voice processing: %s", str(e))
        return None

    finally:
        # Clean up any temporary files that might have been created
        logger.debug("🧹 Cleaning up %s temporary voice files...", len(temp_files))
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug("Deleted temp voice file: %s", temp_file)
                except Exception as cleanup_error:
                    logger.debug("Failed to delete temp voice file %s: %s", temp_file, str(cleanup_error))
