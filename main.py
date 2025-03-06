"""
Audio Transcription Tool (PySide6 + moviepy + GCP Speech-to-Text v2)

Local Environment Settings:
- The file env.txt stores the following settings:
  GOOGLE_APPLICATION_CREDENTIALS=<path>
  GCS_INPUT_BUCKET=<your input bucket name>
  GCS_OUTPUT_BUCKET=<your output bucket name>

Speech API:
- Uses the asynchronous (batch_recognize) v2 API.
- The converted WAV file is uploaded to GCS and then processed.
"""

import sys
import os
import re
import uuid

# Detect if we're running in a frozen environment (PyInstaller).
IS_FROZEN = getattr(sys, 'frozen', False)

# Only check/install dependencies if NOT frozen.
if not IS_FROZEN:
    import subprocess
    missing = []
    try:
        import moviepy
    except ImportError:
        missing.append("moviepy")
    try:
        import imageio_ffmpeg
    except ImportError:
        missing.append("imageio-ffmpeg")
    try:
        from google.cloud import speech_v2
    except ImportError:
        missing.append("google-cloud-speech")
    try:
        from google.cloud import storage
    except ImportError:
        missing.append("google-cloud-storage")
    try:
        from google.oauth2 import service_account
    except ImportError:
        missing.append("google-auth")
    if missing:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        QMessageBox.information(
            None,
            "Dependencies Installed",
            "The following dependencies were installed:\n" + ", ".join(missing) +
            "\nPlease restart the application."
        )
        sys.exit(0)


def load_env():
    """
    Load environment variables from env.txt.
    Expects each line in KEY=VALUE format.
    """
    env_file = "env.txt"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def save_env(text: str):
    """
    Save the provided text to env.txt and reload the environment variables.
    """
    env_file = "env.txt"
    with open(env_file, "w") as f:
        f.write(text)
    load_env()


# Load env on startup.
load_env()

# Now import required PySide6 modules.
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel, QMessageBox,
    QDialog, QDialogButtonBox
)
from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    """
    Worker thread:
    - Converts the input (audio or video) file to a 16 kHz PCM WAV file using moviepy.
    - Uploads the converted file to the GCS input bucket.
    - Calls the Speech-to-Text v2 batch API using the GCS URI and an output location.
    - Downloads and writes the transcript to a local text file.

    Required environment variables:
      GOOGLE_APPLICATION_CREDENTIALS, GCS_INPUT_BUCKET, GCS_OUTPUT_BUCKET
    """
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_file: str, parent=None):
        super().__init__(parent)
        self.input_file = input_file

    def run(self):
        try:
            import imageio_ffmpeg
            import moviepy as mpy
            from google.cloud import speech_v2
            from google.cloud.speech_v2.types import cloud_speech
            from google.cloud import storage
            from google.oauth2 import service_account

            base, ext = os.path.splitext(self.input_file)
            ext = ext.lower()
            temp_audio_file = base + "_temp.wav"

            # Supported file extensions.
            video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
            audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}

            self.progress.emit(5)
            ffmpeg_params = ["-preset", "ultrafast"]

            if ext in video_extensions:
                clip = mpy.VideoFileClip(self.input_file)
                clip.audio.write_audiofile(
                    temp_audio_file, fps=16000, codec='pcm_s16le',
                    logger=None, ffmpeg_params=ffmpeg_params
                )
                clip.close()
            elif ext in audio_extensions:
                clip = mpy.AudioFileClip(self.input_file)
                clip.write_audiofile(
                    temp_audio_file, fps=16000, codec='pcm_s16le',
                    logger=None, ffmpeg_params=ffmpeg_params
                )
                clip.close()
            else:
                self.error.emit("Unsupported file type. Please select a valid audio or video file.")
                return
            self.progress.emit(33)

            # Load credentials from JSON.
            credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credential_path or not os.path.exists(credential_path):
                self.error.emit("Missing or invalid GCP credentials file.")
                return

            credentials = service_account.Credentials.from_service_account_file(credential_path)
            PROJECT_ID = credentials.project_id

            # Verify required GCS environment variables.
            gcs_input_bucket = "speech_to_text_python"
            gcs_output_bucket = "speech_to_text_python"
            if not gcs_input_bucket or not gcs_output_bucket:
                self.error.emit("Missing GCS_INPUT_BUCKET or GCS_OUTPUT_BUCKET settings in env.txt.")
                return

            # Upload the converted audio file to the input bucket.
            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(gcs_input_bucket)
            unique_filename = f"{os.path.basename(temp_audio_file)}-{uuid.uuid4().hex}.wav"
            blob = bucket.blob(unique_filename)
            blob.upload_from_filename(temp_audio_file)
            audio_uri = f"gs://{gcs_input_bucket}/{unique_filename}"
            self.progress.emit(40)

            # Define an output URI for transcription results.
            output_filename = f"transcript-{uuid.uuid4().hex}.json"
            gcs_output_uri = f"gs://{gcs_output_bucket}/{output_filename}"
            self.progress.emit(42)

            # Build the BatchRecognizeRequest.
            file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=audio_uri)
            request = {
                "recognizer": f"projects/{PROJECT_ID}/locations/global/recognizers/_",
                "config": cloud_speech.RecognitionConfig(
                    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                    language_codes=["id-ID"],
                    model="long",
                ),
                "files": [file_metadata],
                "recognition_output_config": cloud_speech.RecognitionOutputConfig(
                    gcs_output_config=cloud_speech.GcsOutputConfig(uri=gcs_output_uri)
                ),
            }

            self.progress.emit(50)
            client = speech_v2.SpeechClient(credentials=credentials)
            operation = client.batch_recognize(request=request)
            self.progress.emit(55)
            response = operation.result(timeout=3600)
            self.progress.emit(75)

            # Retrieve the results.
            file_results = response.results.get(audio_uri)
            if not file_results:
                self.error.emit("No results found for the input file.")
                return

            m = re.match(r"gs://([^/]+)/(.*)", file_results.uri)
            if not m:
                self.error.emit("Failed to parse output URI.")
                return
            output_bucket_name, output_object = m.group(1, 2)
            output_bucket_obj = storage_client.bucket(output_bucket_name)
            output_blob = output_bucket_obj.blob(output_object)
            results_bytes = output_blob.download_as_bytes()
            batch_results = cloud_speech.BatchRecognizeResults.from_json(results_bytes, ignore_unknown_fields=True)

            transcript = ""
            for result in batch_results.results:
                transcript += result.alternatives[0].transcript + "\n"

            self.progress.emit(90)
            output_file = base + ".txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            self.progress.emit(100)

            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            self.finished.emit(f"Transcription saved to {output_file}")

        except Exception as e:
            self.error.emit(str(e))


class EnvironmentDialog(QDialog):
    """
    Environment Settings Dialog:
    Allows the user to set:
      - GOOGLE_APPLICATION_CREDENTIALS: Path to the service account JSON file.
      - GCS_INPUT_BUCKET: Name of the Cloud Storage bucket for uploading audio.
      - GCS_OUTPUT_BUCKET: Name of the Cloud Storage bucket for transcription results.

    The values are saved locally to env.txt.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Environment Settings")
        self.resize(400, 200)

        # Create line edits for each variable.
        self.creds_edit = QLineEdit(self)
        self.creds_edit.setPlaceholderText("Path to GCP Credentials JSON file")
        self.creds_edit.setReadOnly(True)

        self.input_bucket_edit = QLineEdit(self)
        self.input_bucket_edit.setPlaceholderText("GCS Input Bucket Name")

        self.output_bucket_edit = QLineEdit(self)
        self.output_bucket_edit.setPlaceholderText("GCS Output Bucket Name")

        # Browse button for credentials.
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_file)

        creds_layout = QHBoxLayout()
        creds_layout.addWidget(self.creds_edit)
        creds_layout.addWidget(self.browse_button)

        # Buttons.
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)

        # Main layout.
        layout = QVBoxLayout()
        layout.addLayout(creds_layout)
        layout.addWidget(self.input_bucket_edit)
        layout.addWidget(self.output_bucket_edit)
        layout.addWidget(button_box)
        self.setLayout(layout)

        # Load existing values from env.txt if available.
        env_file = "env.txt"
        creds_value = ""
        input_bucket_value = ""
        output_bucket_value = ""
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_APPLICATION_CREDENTIALS="):
                        _, val = line.split("=", 1)
                        creds_value = val.strip()
                    elif line.startswith("GCS_INPUT_BUCKET="):
                        _, val = line.split("=", 1)
                        input_bucket_value = val.strip()
                    elif line.startswith("GCS_OUTPUT_BUCKET="):
                        _, val = line.split("=", 1)
                        output_bucket_value = val.strip()
        self.creds_edit.setText(creds_value)
        self.input_bucket_edit.setText(input_bucket_value)
        self.output_bucket_edit.setText(output_bucket_value)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GCP Credentials JSON file",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.creds_edit.setText(file_path)

    def save_settings(self):
        creds_value = self.creds_edit.text().strip()
        input_bucket_value = self.input_bucket_edit.text().strip()
        output_bucket_value = self.output_bucket_edit.text().strip()
        content = (
            f"GOOGLE_APPLICATION_CREDENTIALS={creds_value}\n"
            f"GCS_INPUT_BUCKET={input_bucket_value}\n"
            f"GCS_OUTPUT_BUCKET={output_bucket_value}\n"
        )
        save_env(content)
        QMessageBox.information(self, "Saved", "Environment settings saved.")
        self.accept()


class MainWindow(QMainWindow):
    """
    Main application window with:
    - File input (for any supported audio/video file)
    - Process button
    - Progress bar
    - Environment settings dialog (to set credentials and bucket names)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Transcription Tool")
        self.worker = None

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Select an audio or video file...")
        self.browse_button = QPushButton("Browse")
        self.process_button = QPushButton("Process")
        self.env_button = QPushButton("Environment Settings")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.input_line)
        file_layout.addWidget(self.browse_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.env_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(file_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.browse_button.clicked.connect(self.browse_file)
        self.process_button.clicked.connect(self.start_processing)
        self.env_button.clicked.connect(self.open_env_dialog)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an audio or video file",
            "",
            "Audio/Video Files (*.mp3 *.wav *.flac *.ogg *.aac *.m4a *.mp4 *.mov *.avi *.mkv *.flv *.wmv)"
        )
        if file_path:
            self.input_line.setText(file_path)
            self.status_label.setText("")

    def start_processing(self):
        file_path = self.input_line.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return

        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")

        self.worker = Worker(file_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error.connect(self.processing_error)
        self.worker.start()

    def processing_finished(self, message: str):
        self.status_label.setText(message)
        QMessageBox.information(self, "Done", message)
        self.process_button.setEnabled(True)

    def processing_error(self, error_message: str):
        self.status_label.setText("Error: " + error_message)
        QMessageBox.critical(self, "Error", error_message)
        self.process_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def open_env_dialog(self):
        dialog = EnvironmentDialog(self)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(500, 300)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
