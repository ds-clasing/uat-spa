# type: ignore
import matplotlib
matplotlib.use('Agg')
import json
import os
import shutil
import re
import logging
import sys
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from g2p_en import G2p
import nltk
import google.generativeai as genai

class ClassroomAudioAnalyzer:
    """
    A framework to analyze classroom audio recordings and transcripts.

    This class encapsulates the entire analysis pipeline, including data preparation,
    pronunciation analysis, vocabulary estimation, and report generation.
    It uses a structured logging system for clear, staged feedback.
    """

    # --- Pronunciation Scoring & Vocabulary Constants ---
    REPETITION_PATTERN = r'\b(\w+)\s+\1\b'
    WEIGHT_WPM = 0.20
    WEIGHT_PAUSE = 0.15
    WEIGHT_DISFLUENCY = 0.15
    WEIGHT_ACCURACY_PROXY = 0.50
    TARGET_WPM = 150
    TARGET_AVG_PAUSE_S = 0.5
    MAX_EXPECTED_DISFLUENCIES_PER_100_WORDS = 5
    MAX_EXPECTED_PAUSES = 10
    CEFR_LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "Unknown"]

    def __init__(self, transcript_file: str, audio_file: str, gemini_api_key: str, output_dir_base: str):
        """
        Initializes the ClassroomAudioAnalyzer.

        Args:
            transcript_file (str): Path to the JSON transcript file.
            audio_file (str): Path to the audio/video file.
            gemini_api_key (str): Your Google Generative AI API key.
            output_dir_base (str): The base name for the output directory.
        """
        self.transcript_file = transcript_file
        self.audio_file = audio_file
        self.gemini_api_key = gemini_api_key

        # --- Configure Paths ---
        self.base_output_dir = output_dir_base
        self.speaker_segments_dir = os.path.join(self.base_output_dir, "speaker_audio_segments")
        self.forced_alignment_dir = os.path.join(self.base_output_dir, "forced_alignment_visuals")
        self.plots_dir = os.path.join(self.base_output_dir, "plots")
        self.reports_dir = os.path.join(self.base_output_dir, "reports")
        self.log_file_path = os.path.join(self.base_output_dir, "analysis_log.txt")

        # --- Analysis Thresholds ---
        self.relevance_threshold_words = 10
        self.relevance_threshold_utterances = 0
        self.long_silence_threshold_ms = 1000
        self.confidence_threshold_for_correct = 0.80
        self.mispronunciation_confidence_threshold = 0.75
        self.min_pause_duration_ms_between_utterances = 500

        # --- Initialize Data Structures ---
        self.transcript_data = None
        self.main_audio = None
        self.aligned_words_segments = []
        self.speaker_utterances_map = {}
        self.speaker_fragments_map = {}
        self.speaker_speaking_time_s_map = {}
        self.student_pronunciation_metrics = {}
        self.cefr_levels_dict = {}
        self.df_mispronounced = pd.DataFrame()
        self.reports = []
        self.g2p = None
        
        # --- Setup Logger ---
        self._setup_logging()

    def _setup_logging(self):
        """Configures the logger to output to console and a file."""
        self._ensure_dir(self.base_output_dir, clear_if_exists=False) # Log file needs the dir
        self.logger = logging.getLogger(f"Analyzer_{self.base_output_dir}")
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if re-initialized
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File Handler
        fh = logging.FileHandler(self.log_file_path)
        fh.setLevel(logging.INFO)
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _ensure_dir(self, directory_path: str, clear_if_exists: bool = True):
        """Ensures a directory exists, optionally clearing it first."""
        if clear_if_exists and os.path.exists(directory_path):
            self.logger.info(f"Directory '{directory_path}' already exists. Clearing it for fresh run.")
            try:
                shutil.rmtree(directory_path)
            except OSError as e:
                self.logger.error(f"Error clearing directory {directory_path}: {e}. Attempting to continue.")
        try:
            os.makedirs(directory_path, exist_ok=True)
            # self.logger.info(f"Ensured directory exists: '{directory_path}'")
        except OSError as e:
            self.logger.critical(f"FATAL: Could not create directory {directory_path}: {e}.")
            sys.exit(1)

    def _initialize_tools(self):
        """Initializes external tools like Gemini, NLTK, and G2P."""
        self.logger.info("--- Initializing External Tools ---")
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.logger.info("Gemini API configured successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to configure Gemini API: {e}")
            sys.exit(1)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng.zip')
        except LookupError:
            self.logger.info("NLTK 'averaged_perceptron_tagger_eng' not found. Downloading...")
            nltk.download('averaged_perceptron_tagger_eng')
        
        self.g2p = G2p()
        self.logger.info("G2P (g2p-en) initialized.")

    def _detect_simple_repetitions(self, text: str) -> int:
        """Detects simple repetitions and common filler words."""
        repetitions = len(re.findall(self.REPETITION_PATTERN, text.lower()))
        fillers = ["um", "uh", "er", "ah", "hmm", "like", "you know", "so"]
        filler_count = 0
        words_in_text = text.lower().split()
        i = 0
        while i < len(words_in_text):
            if i < len(words_in_text) - 1 and f"{words_in_text[i]} {words_in_text[i+1]}" in fillers:
                filler_count += 1
                i += 2
            elif words_in_text[i] in fillers:
                filler_count += 1
                i += 1
            else:
                i += 1
        return repetitions + filler_count

    def _get_cefr_levels_with_gemini(self, words_list: list, batch_size: int = 200) -> dict:
        """Gets CEFR levels for a list of words using the Gemini API."""
        if not words_list:
            return {}

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        cefr_dict = {}

        self.logger.info(f"Starting CEFR classification for {len(words_list)} unique words using Gemini...")

        for i in range(0, len(words_list), batch_size):
            batch = words_list[i:i + batch_size]
            self.logger.info(f"  - Processing batch {i//batch_size + 1}...")

            words_string = ", ".join(f'"{word}"' for word in batch)
            prompt = f"""
            Analyze the following list of English words and classify each one according to the
            Common European Framework of Reference for Languages (CEFR).

            Your task is to return a single, valid JSON object.
            - The keys of the JSON object should be the words from the list.
            - The values should be their corresponding CEFR level as a string: "A1", "A2", "B1", "B2", "C1", or "C2".
            - If a word is a proper noun (like a name or city), a brand, an acronym, a non-English word, or cannot be classified,
              use the value "Unknown".

            Example Request Words: ["apple", "hello", "sophisticated", "Barcelona", "photosynthesis", "asdfg"]
            Example JSON Response:
            {{
              "apple": "A1",
              "hello": "A1",
              "sophisticated": "C1",
              "Barcelona": "Unknown",
              "photosynthesis": "C2",
              "asdfg": "Unknown"
            }}

            Now, please classify the following words:
            [{words_string}]
            """

            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
                )
                batch_result = json.loads(response.text)
                cefr_dict.update(batch_result)
            except Exception as e:
                self.logger.error(f"An error occurred during Gemini API call for batch {i//batch_size + 1}: {e}")
                for word in batch:
                    cefr_dict[word] = "Unknown"

        self.logger.info(f"CEFR classification complete. {len(cefr_dict)} words classified.")
        return cefr_dict

    def _load_data(self):
        """Loads the transcript and audio files."""
        self.logger.info("--- Step 0: Setup and Data Loading ---")
        self._ensure_dir(self.speaker_segments_dir)
        self._ensure_dir(self.plots_dir)
        self._ensure_dir(self.reports_dir)
        
        try:
            with open(self.transcript_file, 'r', encoding='utf-8') as f:
                self.transcript_data = json.load(f)
            self.logger.info(f"Successfully loaded transcript data. Found {len(self.transcript_data.get('words', []))} words and {len(self.transcript_data.get('utterances', []))} utterances.")
        except Exception as e:
            self.logger.critical(f"FATAL: Could not load transcript file '{self.transcript_file}': {e}")
            sys.exit(1)

        try:
            # Assuming audio file format can be inferred, common formats like mp4, wav, mp3 are handled by pydub with ffmpeg
            file_extension = os.path.splitext(self.audio_file)[1].replace('.', '')
            self.main_audio = AudioSegment.from_file(self.audio_file, format=file_extension)
            self.logger.info(f"Successfully loaded audio file '{self.audio_file}'.")
        except Exception as e:
            self.logger.warning(f"Could not load audio file '{self.audio_file}': {e}. Audio segmentation will be simulated.")

    def _prepare_data(self):
        """
        Runs the entire data preparation pipeline (Step 1).
        Includes a weighted mechanism to determine speaker relevance for multi-speaker recordings.
        """
        self.logger.info("\n--- Step 1: Data Preparation ---")

        # 1.1 Forced Alignment (from transcript data)
        self.logger.info("#### 1.1 Forced Alignment (from transcript data) ####")
        if 'words' in self.transcript_data:
            for word_info in self.transcript_data['words']:
                self.aligned_words_segments.append({
                    'text': word_info.get('text', ''),
                    'start_ms': word_info.get('start', 0),
                    'end_ms': word_info.get('end', 0),
                    'speaker': word_info.get('speaker', 'Unknown'),
                    'confidence': word_info.get('confidence', 0.0)
                })
        self.logger.info(f"Processed {len(self.aligned_words_segments)} aligned word segments.")

        # --- New Speaker Relevance Logic ---
        self.logger.info("#### Determining Relevant Speakers ####")
        all_utterances = self.transcript_data.get('utterances', [])
        if not all_utterances:
            self.logger.warning("No utterances found in transcript data. Cannot proceed with analysis.")
            return

        all_speakers = set(utt.get('speaker', 'Unknown') for utt in all_utterances)
        relevant_speakers = set()

        # Condition: Apply weighted analysis only if more than 2 speakers are present
        if len(all_speakers) > 2:
            self.logger.info(f"Found {len(all_speakers)} speakers. Applying weighted relevance analysis to identify key participants.")

            # 1. Calculate metrics for each speaker from all utterances
            speaker_word_count = defaultdict(int)
            speaker_speaking_time_ms = defaultdict(int)
            speaker_utterance_count = defaultdict(int)

            for utt in all_utterances:
                speaker = utt.get('speaker', 'Unknown')
                start_ms = utt.get('start', 0)
                end_ms = utt.get('end', 0)
                
                speaker_utterance_count[speaker] += 1
                speaker_speaking_time_ms[speaker] += (end_ms - start_ms)
                speaker_word_count[speaker] += len(utt.get('words', []))

            # 2. Calculate totals for normalization
            total_words = sum(speaker_word_count.values())
            total_time_ms = sum(speaker_speaking_time_ms.values())
            total_utterances = sum(speaker_utterance_count.values())

            # Handle case with no data to prevent division by zero
            if total_words == 0 and total_time_ms == 0 and total_utterances == 0:
                self.logger.warning("No words, speaking time, or utterances found. Cannot perform relevance analysis. Treating all speakers as relevant.")
                relevant_speakers = all_speakers
            else:
                # 3. Define weights and calculate scores
                W_WORDS = 0.50
                W_TIME = 0.30
                W_UTTERANCES = 0.20
                
                speaker_scores = {}
                for speaker in all_speakers:
                    norm_words = (speaker_word_count[speaker] / total_words) if total_words > 0 else 0
                    norm_time = (speaker_speaking_time_ms[speaker] / total_time_ms) if total_time_ms > 0 else 0
                    norm_utterances = (speaker_utterance_count[speaker] / total_utterances) if total_utterances > 0 else 0
                    
                    score = (W_WORDS * norm_words) + (W_TIME * norm_time) + (W_UTTERANCES * norm_utterances)
                    speaker_scores[speaker] = score

                # 4. Determine relevance based on a dynamic threshold
                # A speaker is relevant if their contribution is at least 25% of an "average" speaker's contribution.
                # An average speaker's score would be roughly 1.0 / len(all_speakers).
                relevance_threshold = (1.0 / len(all_speakers)) * 0.25
                
                self.logger.info(f"Speaker Relevance Scoring (Threshold={relevance_threshold:.4f}, Weights: Words={W_WORDS*100}%, Time={W_TIME*100}%, Utterances={W_UTTERANCES*100}%):")
                for speaker, score in sorted(speaker_scores.items(), key=lambda item: item[1], reverse=True):
                    is_relevant = score >= relevance_threshold
                    if is_relevant:
                        relevant_speakers.add(speaker)
                    status = "Relevant" if is_relevant else "Not Relevant"
                    self.logger.info(f"  - {speaker}: Score={score:.4f} --> {status}")

        else:
            self.logger.info(f"Found {len(all_speakers)} speaker(s). Using simple utterance count filter (Threshold: >= {self.relevance_threshold_utterances} utterances).")
            # Use the original, simpler logic for 2 or fewer speakers
            utterance_count_map = defaultdict(int)
            for utt in all_utterances:
                utterance_count_map[utt.get('speaker', 'Unknown')] += 1

            relevant_speakers = {speaker for speaker, count in utterance_count_map.items() if count >= self.relevance_threshold_utterances}
            self.logger.info("Speaker Relevance Status:")
            for speaker in all_speakers:
                status = "Relevant" if speaker in relevant_speakers else "Not Relevant"
                self.logger.info(f"  - {speaker}: {utterance_count_map.get(speaker, 0)} utterances --> {status}")

        # --- Filter data based on the determined relevant speakers ---
        aligned_words_segments_original_count = len(self.aligned_words_segments)
        self.aligned_words_segments = [seg for seg in self.aligned_words_segments if seg['speaker'] in relevant_speakers]
        self.logger.info(f"Remaining segments after filtering irrelevant speakers: {len(self.aligned_words_segments)} (from {aligned_words_segments_original_count})")
        
        unique_speakers_final = set(seg['speaker'] for seg in self.aligned_words_segments)
        self.logger.info(f"Final set of unique, relevant speakers for analysis: {unique_speakers_final}")
        
        if self.aligned_words_segments:
            num_words_with_confidence = sum(1 for s in self.aligned_words_segments if s['confidence'] is not None)
            if num_words_with_confidence > 0:
                total_confidence = sum(s['confidence'] for s in self.aligned_words_segments if s['confidence'] is not None)
                avg_confidence = total_confidence / num_words_with_confidence
                self.logger.info(f"Average word confidence across all relevant speakers: {avg_confidence:.3f}")

        # 1.2 Speaker Diarization and Audio Segmentation
        self.logger.info("#### 1.2 - Speaker Diarization and Audio Segmentation ####")
        _speaker_utterances_temp = defaultdict(list)
        for i, utterance in enumerate(self.transcript_data['utterances']):
            speaker = utterance.get('speaker', f'UnknownSpeaker_{i}')
            if speaker not in relevant_speakers:
                continue

            start_ms, end_ms, text = utterance.get('start', 0), utterance.get('end', 0), utterance.get('text', '')
            utterance_details = {'id': f"utt_{speaker}_{len(_speaker_utterances_temp[speaker])}", 'text': text, 'start_ms': start_ms, 'end_ms': end_ms, 'duration_ms': end_ms - start_ms, 'words': utterance.get('words', []), 'audio_segment_path': None}

            if self.main_audio:
                try:
                    segment = self.main_audio[start_ms:end_ms]
                    segment_filename = os.path.join(self.speaker_segments_dir, f"speaker_{speaker}_utt_{len(_speaker_utterances_temp[speaker])}.wav")
                    segment.export(segment_filename, format="wav")
                    utterance_details['audio_segment_path'] = segment_filename
                except Exception as e:
                    self.logger.error(f"Error segmenting/exporting audio for utterance {i} (Speaker {speaker}): {e}")
                    utterance_details['audio_segment_path'] = "ERROR_EXPORTING"
            else:
                utterance_details['audio_segment_path'] = "SIMULATED_AUDIO"
            _speaker_utterances_temp[speaker].append(utterance_details)
        
        self.speaker_utterances_map = dict(_speaker_utterances_temp)
        self.logger.info(f"Speaker diarization complete. Identified and processed relevant speakers: {list(self.speaker_utterances_map.keys())}")

        # 1.3 Division into Relevant Fragments & Speaking Time
        self.logger.info("#### 1.3 - Division into Relevant Fragments & Speaking Time ####")
        for speaker, utterances in self.speaker_utterances_map.items():
            self.speaker_fragments_map[speaker] = {'phrases': [], 'words': []}
            current_speaker_total_word_duration_ms = 0
            for utterance in utterances:
                self.speaker_fragments_map[speaker]['phrases'].append({'id': utterance['id'], 'text': utterance['text'], 'start_ms': utterance['start_ms'], 'end_ms': utterance['end_ms'], 'duration_ms': utterance['duration_ms'], 'audio_segment_path': utterance['audio_segment_path']})
                if 'words' in utterance and utterance['words']:
                    for word_info in utterance['words']:
                        word_start, word_end = word_info.get('start', 0), word_info.get('end', 0)
                        self.speaker_fragments_map[speaker]['words'].append({'text': word_info.get('text', ''), 'start_ms': word_start, 'end_ms': word_end, 'duration_ms': word_end - word_start, 'parent_utterance_id': utterance['id']})
                        current_speaker_total_word_duration_ms += (word_end - word_start)
                else:
                    current_speaker_total_word_duration_ms += utterance['duration_ms']
            self.speaker_speaking_time_s_map[speaker] = current_speaker_total_word_duration_ms / 1000.0
            self.logger.info(f"Speaker {speaker}: Total speaking time: {self.speaker_speaking_time_s_map[speaker]:.2f}s, Phrases: {len(self.speaker_fragments_map[speaker]['phrases'])}, Words: {len(self.speaker_fragments_map[speaker]['words'])}")

        # 1.4 Silence Detection
        self.logger.info("#### 1.4 - Silence Detection ####")
        silences_found = defaultdict(list)
        speaker_words_chronological = defaultdict(list)
        for word_seg in self.aligned_words_segments:
            speaker_words_chronological[word_seg['speaker']].append(word_seg)
        for speaker, words_list in speaker_words_chronological.items():
            sorted_words = sorted(words_list, key=lambda x: x['start_ms'])
            for i in range(len(sorted_words) - 1):
                current_word, next_word = sorted_words[i], sorted_words[i+1]
                silence_duration_ms = next_word['start_ms'] - current_word['end_ms']
                if silence_duration_ms >= self.long_silence_threshold_ms:
                    silences_found[speaker].append({'after_word': current_word['text'], 'before_word': next_word['text'], 'duration_ms': silence_duration_ms})
            self.logger.info(f"Speaker {speaker}: Found {len(silences_found[speaker])} long silences (>= {self.long_silence_threshold_ms}ms).")

    def _generate_preliminary_plots(self):
        """Generates and saves all plots related to Step 1."""
        self.logger.info("\n--- Generating Step 1 Summary Plots ---")
        sns.set_theme(style="whitegrid")
        # Plot 1: Speaking Time
        if self.speaker_speaking_time_s_map:
            plt.figure(figsize=(8, 5)); barplot = sns.barplot(x=list(self.speaker_speaking_time_s_map.keys()), y=list(self.speaker_speaking_time_s_map.values()), palette="viridis", hue=list(self.speaker_speaking_time_s_map.keys()), dodge=False, legend=False); plt.title('Total Speaking Time per Speaker'); plt.xlabel('Speaker'); plt.ylabel('Speaking Time (seconds)'); [barplot.text(i, v + 0.5, f"{v:.1f}s", color='black', ha="center") for i, v in enumerate(self.speaker_speaking_time_s_map.values())]; plot_path = os.path.join(self.plots_dir, "plot_1_speaking_time.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        # Plot 2: Utterance Count
        num_utterances_data = {s: len(u) for s, u in self.speaker_utterances_map.items()}
        if num_utterances_data:
            plt.figure(figsize=(8, 5)); barplot = sns.barplot(x=list(num_utterances_data.keys()), y=list(num_utterances_data.values()), palette="mako", hue=list(num_utterances_data.keys()), dodge=False, legend=False); plt.title('Number of Utterances (Turns) per Speaker'); plt.xlabel('Speaker'); plt.ylabel('Number of Utterances'); [barplot.text(i, v + (0.01 * max(num_utterances_data.values())), str(v), color='black', ha="center") for i, v in enumerate(num_utterances_data.values())]; plot_path = os.path.join(self.plots_dir, "plot_2_utterance_count.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        # Plot 3: Utterance Duration
        speaker_duration_data = [{'speaker': s, 'duration_s': u['duration_ms'] / 1000.0} for s, ul in self.speaker_utterances_map.items() for u in ul]
        if speaker_duration_data:
            df_durations = pd.DataFrame(speaker_duration_data); plt.figure(figsize=(10, 6)); sns.boxplot(x='speaker', y='duration_s', data=df_durations, palette="pastel", hue='speaker', legend=False); plt.title('Distribution of Utterance Durations per Speaker'); plt.ylim(0, df_durations['duration_s'].quantile(0.95) * 1.1); plot_path = os.path.join(self.plots_dir, "plot_3_utterance_duration.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        # Plot 4: Word Confidence
        speaker_confidence_data = [{'speaker': seg['speaker'], 'confidence': seg['confidence']} for seg in self.aligned_words_segments if seg['confidence'] is not None]
        if speaker_confidence_data:
            df_confidences = pd.DataFrame(speaker_confidence_data); plt.figure(figsize=(10, 6)); sns.boxplot(x='speaker', y='confidence', data=df_confidences, palette="coolwarm", hue='speaker', legend=False); plt.title('Distribution of Word Confidence Scores per Speaker'); plt.ylim(min(0.5, df_confidences['confidence'].min()), 1.0); plot_path = os.path.join(self.plots_dir, "plot_4_word_confidence.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        # Plot 5: WPM
        wpm_data = {s: (len(fmap['words']) / t_sec * 60 if t_sec > 0 else 0) for s, fmap, t_sec in zip(self.speaker_fragments_map.keys(), self.speaker_fragments_map.values(), self.speaker_speaking_time_s_map.values())}
        if wpm_data:
            plt.figure(figsize=(8, 5)); barplot = sns.barplot(x=list(wpm_data.keys()), y=list(wpm_data.values()), palette="crest", hue=list(wpm_data.keys()), dodge=False, legend=False); plt.title('Estimated Speaking Pace per Speaker (WPM)'); [barplot.text(i, v + 2, f"{v:.0f}", color='black', ha="center") for i, v in enumerate(wpm_data.values())]; plot_path = os.path.join(self.plots_dir, "plot_5_wpm.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        # Plot 6: Timeline
        timeline_data = [{'Speaker': s, 'Start (s)': u['start_ms']/1000.0, 'End (s)': u['end_ms']/1000.0, 'Duration (s)': u['duration_ms']/1000.0} for s, ul in self.speaker_utterances_map.items() for u in ul]
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data); unique_speakers = df_timeline['Speaker'].unique(); palette_choice = sns.color_palette("husl", len(unique_speakers)); speaker_color_map = {speaker: palette_choice[i % len(palette_choice)] for i, speaker in enumerate(unique_speakers)}; fig, ax = plt.subplots(figsize=(25, max(4, len(unique_speakers) * 0.8))); speaker_y_pos = {speaker: i for i, speaker in enumerate(unique_speakers)}; [ax.barh(y=speaker_y_pos[row['Speaker']], width=row['Duration (s)'], left=row['Start (s)'], height=0.6, color=speaker_color_map[row['Speaker']], edgecolor="black") for _, row in df_timeline.iterrows()]; ax.set_yticks(list(speaker_y_pos.values())); ax.set_yticklabels(list(speaker_y_pos.keys())); ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Speaker"); plt.title("Speaker Activity Timeline (Utterances)"); max_time = df_timeline['End (s)'].max(); ax.set_xlim(0, max_time + 10); plt.grid(True, axis='x', linestyle=':', alpha=0.7); plt.tight_layout(); plot_path = os.path.join(self.plots_dir, "plot_6_timeline.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
    
    def _analyze_pronunciation(self):
        """Runs the entire pronunciation analysis pipeline (Step 2)."""
        self.logger.info("\n--- Step 2: Pronunciation Analysis ---")
        
        # 2.2.1 Computing Fluency Metrics
        self.logger.info("#### 2.2.1: Computing Fluency Metrics ####")
        for speaker in self.speaker_utterances_map.keys():
            self.student_pronunciation_metrics.setdefault(speaker, {})
            num_words = len(self.speaker_fragments_map.get(speaker, {}).get('words', []))
            speaking_time_sec = self.speaker_speaking_time_s_map.get(speaker, 0)
            self.student_pronunciation_metrics[speaker]['wpm'] = round((num_words / speaking_time_sec) * 60 if speaking_time_sec > 0 else 0, 2)
            
            utterances = sorted(self.speaker_utterances_map[speaker], key=lambda x: x['start_ms'])
            pause_durations_ms = [u2['start_ms'] - u1['end_ms'] for u1, u2 in zip(utterances, utterances[1:]) if u2['start_ms'] - u1['end_ms'] >= self.min_pause_duration_ms_between_utterances]
            avg_pause_s = (sum(pause_durations_ms) / len(pause_durations_ms)) / 1000.0 if pause_durations_ms else 0
            self.student_pronunciation_metrics[speaker]['avg_pause_s_between_utterances'] = round(avg_pause_s, 2)
            self.student_pronunciation_metrics[speaker]['num_long_pauses_between_utterances'] = len(pause_durations_ms)
            
            full_text = " ".join([utt['text'] for utt in utterances])
            self.student_pronunciation_metrics[speaker]['disfluency_count'] = self._detect_simple_repetitions(full_text)
            self.logger.info(f"Speaker {speaker} Fluency: WPM={self.student_pronunciation_metrics[speaker]['wpm']}, Avg Pause={avg_pause_s:.2f}s, Disfluencies={self.student_pronunciation_metrics[speaker]['disfluency_count']}")

        # 2.2.2 Computing Accuracy/Clearness Metrics (Proxy)
        self.logger.info("#### 2.2.2: Computing Accuracy/Clearness Metrics (Proxy) ####")
        for speaker in sorted(list(set(seg['speaker'] for seg in self.aligned_words_segments))):
            self.student_pronunciation_metrics.setdefault(speaker, {})
            speaker_words = [s for s in self.aligned_words_segments if s['speaker'] == speaker and s['confidence'] is not None]
            if not speaker_words:
                self.logger.warning(f"Speaker {speaker}: No word data with confidence found. Skipping accuracy metrics.")
                continue
            confidences = [w['confidence'] for w in speaker_words]
            avg_dev_proxy = sum(1 - c for c in confidences) / len(confidences) if confidences else 0
            self.student_pronunciation_metrics[speaker]['avg_phonetic_deviation_proxy'] = round(avg_dev_proxy, 3)
            correct_count = sum(1 for w in speaker_words if w['confidence'] >= self.confidence_threshold_for_correct)
            percent_correct = (correct_count / len(speaker_words)) * 100 if speaker_words else 0
            self.student_pronunciation_metrics[speaker]['percent_words_correct_proxy'] = round(percent_correct, 2)
            self.student_pronunciation_metrics[speaker]['mispronounced_words_proxy'] = sorted([{'text': w['text'], 'confidence': round(w['confidence'], 2)} for w in speaker_words if w['confidence'] < self.confidence_threshold_for_correct], key=lambda x: x['confidence'])[:10]
            self.logger.info(f"Speaker {speaker} Accuracy (Proxy): Avg. Deviation={avg_dev_proxy:.3f}, % Correct={percent_correct:.2f}%")

        # 2.2.3 Computing Pronunciation Score (Heuristic)
        self.logger.info("#### 2.2.3: Computing Pronunciation Score (Heuristic) ####")
        for speaker, metrics in self.student_pronunciation_metrics.items():
            wpm_score = min(metrics.get('wpm', 0) / self.TARGET_WPM, 1.0) * 100 * self.WEIGHT_WPM
            pause_score = max(0, 1 - (metrics.get('num_long_pauses_between_utterances', 0) / self.MAX_EXPECTED_PAUSES)) * 100 * self.WEIGHT_PAUSE
            num_words = len(self.speaker_fragments_map.get(speaker, {}).get('words', []))
            disfluency_rate = (metrics.get('disfluency_count', 0) / num_words) * 100 if num_words > 0 else 0
            disfluency_score = max(0, 1 - (disfluency_rate / self.MAX_EXPECTED_DISFLUENCIES_PER_100_WORDS)) * 100 * self.WEIGHT_DISFLUENCY
            accuracy_score = metrics.get('percent_words_correct_proxy', 0) * self.WEIGHT_ACCURACY_PROXY
            total_score = wpm_score + pause_score + disfluency_score + accuracy_score
            self.student_pronunciation_metrics[speaker]['overall_pronunciation_score_heuristic'] = round(total_score, 2)
            self.logger.info(f"Speaker {speaker}: Overall Heuristic Pronunciation Score: {total_score:.2f} / 100")
        
        # Plot 7: Heuristic Scores
        scores_to_plot = [{'Speaker': s, 'Score': m.get('overall_pronunciation_score_heuristic', 0)} for s, m in self.student_pronunciation_metrics.items()]
        if scores_to_plot:
            df_scores = pd.DataFrame(scores_to_plot); plt.figure(figsize=(8, 5)); barplot = sns.barplot(x='Speaker', y='Score', data=df_scores, palette="magma", hue='Speaker', dodge=False, legend=False); plt.title('Overall Heuristic Pronunciation Score per Speaker'); plt.ylabel('Score (out of 100)'); plt.ylim(0, 105); [barplot.text(bar_obj.get_x() + bar_obj.get_width() / 2., bar_obj.get_height() + 1, f"{bar_obj.get_height():.1f}", ha='center', va='bottom') for bar_obj in barplot.patches]; plot_path = os.path.join(self.plots_dir, "plot_7_heuristic_scores.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")

    # def _identify_specific_errors(self):
    #     """Identifies and records specific word pronunciation errors (Step 3)."""
    #     self.logger.info("\n--- Step 3: Identification of Specific Errors ---")
        
    #     mispronounced_word_details_list = []
    #     self.logger.info("#### 3.1 & 3.2: Detecting and Recording (Potentially) Mispronounced Words ####")
    #     for word_info in self.aligned_words_segments:
    #         word_text_cleaned = re.sub(r'[^\w\s]', '', word_info.get('text', '').strip().lower())
    #         if not word_text_cleaned: continue
    #         confidence = word_info.get('confidence', 1.0)
    #         if confidence < self.mispronunciation_confidence_threshold:
    #             standard_phonemes = ["N/A"]
    #             try: standard_phonemes = self.g2p(word_text_cleaned)
    #             except Exception: pass
                
    #             audio_clip_path = "N/A"
    #             for utt_s, utts in self.speaker_utterances_map.items():
    #                 if utt_s == word_info['speaker']:
    #                     for utt in utts:
    #                         if utt['start_ms'] <= word_info['start_ms'] and utt['end_ms'] >= word_info['end_ms']:
    #                             audio_clip_path = utt['audio_segment_path']; break
    #                     if audio_clip_path != "N/A": break

    #             mispronounced_word_details_list.append({'speaker': word_info['speaker'], 'expected_word': word_text_cleaned, 'standard_pronunciation_ipa': " ".join(standard_phonemes), 'start_ms_in_full_audio': word_info['start_ms'], 'end_ms_in_full_audio': word_info['end_ms'], 'audio_clip_path_utterance': audio_clip_path, 'word_pronunciation_score_proxy': round(confidence * 100, 1), 'asr_confidence': round(confidence, 3), 'lexical_complexity': self.cefr_levels_dict.get(word_text_cleaned, "Unknown")})
        
    #     self.logger.info(f"Processed all words. Found {len(mispronounced_word_details_list)} potentially mispronounced words.")
    #     self.df_mispronounced = pd.DataFrame(mispronounced_word_details_list)
    #     mispronounced_csv_path = os.path.join(self.reports_dir, "potentially_mispronounced_words.csv")
    #     self.df_mispronounced.to_csv(mispronounced_csv_path, index=False)
    #     self.logger.info(f"Full list of potentially mispronounced words saved to: {mispronounced_csv_path}")

    #     # Plot 8: Mispronounced Count
    #     if not self.df_mispronounced.empty:
    #         plt.figure(figsize=(10, 6)); sns.countplot(data=self.df_mispronounced, x='speaker', palette="Set2", hue='speaker', legend=False); plt.title(f'Count of Potentially Mispronounced Words (Confidence < {self.mispronunciation_confidence_threshold})'); plt.ylabel('Number of Potentially Mispronounced Words'); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plot_path = os.path.join(self.plots_dir, "plot_8_mispronounced_count.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
    #     # Plot 9: Confidence Distribution
    #     if not self.df_mispronounced.empty:
    #         plt.figure(figsize=(10, 5)); sns.histplot(self.df_mispronounced['asr_confidence'], kde=True, bins=15, color='salmon'); plt.title(f'ASR Confidence Distribution for Flagged Words (Threshold < {self.mispronunciation_confidence_threshold})'); plt.xlim(0, self.mispronunciation_confidence_threshold); plot_path = os.path.join(self.plots_dir, "plot_9_mispronounced_confidence.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")

    def _identify_specific_errors(self):
        """Identifies and records specific word pronunciation errors (Step 3)."""
        self.logger.info("\n--- Step 3: Identification of Specific Errors ---")
        
        # --- NEW: Create a dedicated directory for word audio clips ---
        word_clips_dir = os.path.join(self.base_output_dir, "word_audio_clips")
        self._ensure_dir(word_clips_dir, clear_if_exists=True) # Use the internal ensure_dir method
        
        mispronounced_word_details_list = []
        self.logger.info("#### 3.1 & 3.2: Detecting, Recording, and Segmenting (Potentially) Mispronounced Words ####")
        
        for i, word_info in enumerate(self.aligned_words_segments):
            word_text_cleaned = re.sub(r'[^\w\s]', '', word_info.get('text', '').strip().lower())
            if not word_text_cleaned: continue
            
            confidence = word_info.get('confidence', 1.0)
            
            if confidence < self.mispronunciation_confidence_threshold:
                standard_phonemes = ["N/A"]
                try: standard_phonemes = self.g2p(word_text_cleaned)
                except Exception: pass
                
                # --- MODIFIED: Export individual word audio clip ---
                word_audio_path = "N/A"
                if self.main_audio and word_info['end_ms'] > word_info['start_ms']:
                    try:
                        # Extract the specific word's audio from the main audio file
                        word_segment = self.main_audio[word_info['start_ms']:word_info['end_ms']]
                        
                        # Create a unique, safe filename
                        word_clip_filename = f"word_{i}_{word_info['speaker']}_{word_text_cleaned}.wav"
                        word_audio_path_full = os.path.join(word_clips_dir, word_clip_filename)
                        
                        word_segment.export(word_audio_path_full, format="wav")
                        
                        # Store the RELATIVE path for the JSON report
                        word_audio_path = os.path.join("word_audio_clips", word_clip_filename)

                    except Exception as e:
                        self.logger.error(f"Could not export audio for word '{word_text_cleaned}': {e}")
                        word_audio_path = "ERROR_EXPORTING"

                mispronounced_word_details_list.append({
                    'speaker': word_info['speaker'],
                    'expected_word': word_text_cleaned,
                    'standard_pronunciation_ipa': " ".join(standard_phonemes),
                    'start_ms_in_full_audio': word_info['start_ms'],
                    'end_ms_in_full_audio': word_info['end_ms'],
                    # 'audio_clip_path_utterance': audio_clip_path, # This is no longer needed
                    'word_audio_clip_path': word_audio_path, # NEW: The path to the specific word's audio
                    'word_pronunciation_score_proxy': round(confidence * 100, 1),
                    'asr_confidence': round(confidence, 3),
                    'lexical_complexity': self.cefr_levels_dict.get(word_text_cleaned, "Unknown")
                })
        
        self.logger.info(f"Processed all words. Found {len(mispronounced_word_details_list)} potentially mispronounced words and exported their audio clips.")
        self.df_mispronounced = pd.DataFrame(mispronounced_word_details_list)
        mispronounced_csv_path = os.path.join(self.reports_dir, "potentially_mispronounced_words.csv")
        self.df_mispronounced.to_csv(mispronounced_csv_path, index=False)
        self.logger.info(f"Full list of potentially mispronounced words saved to: {mispronounced_csv_path}")

        # --- The rest of the method (plotting) remains the same ---
        if not self.df_mispronounced.empty:
            plt.figure(figsize=(10, 6)); sns.countplot(data=self.df_mispronounced, x='speaker', palette="Set2", hue='speaker', legend=False); plt.title(f'Count of Potentially Mispronounced Words (Confidence < {self.mispronunciation_confidence_threshold})'); plt.ylabel('Number of Potentially Mispronounced Words'); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plot_path = os.path.join(self.plots_dir, "plot_8_mispronounced_count.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")
        if not self.df_mispronounced.empty:
            plt.figure(figsize=(10, 5)); sns.histplot(self.df_mispronounced['asr_confidence'], kde=True, bins=15, color='salmon'); plt.title(f'ASR Confidence Distribution for Flagged Words (Threshold < {self.mispronunciation_confidence_threshold})'); plt.xlim(0, self.mispronunciation_confidence_threshold); plot_path = os.path.join(self.plots_dir, "plot_9_mispronounced_confidence.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")

    def _estimate_vocabulary_level(self):
        """Estimates vocabulary level for each speaker (Step 4)."""
        self.logger.info("\n--- Step 4: Vocabulary Level Estimation ---")
        
        # Dynamically build the CEFR dictionary
        all_words_from_transcript = [seg['text'] for seg in self.aligned_words_segments]
        cleaned_unique_words = sorted(list(set([re.sub(r'[^\w\s-]', '', w.lower()) for w in all_words_from_transcript if w and len(w) > 1])))
        self.cefr_levels_dict = self._get_cefr_levels_with_gemini(cleaned_unique_words)
        if not self.cefr_levels_dict:
            self.logger.warning("Gemini call failed or returned no data. Vocabulary analysis will be limited.")

        speaker_vocabulary_data = {}
        for speaker in self.speaker_fragments_map.keys():
            words = [w['text'] for w in self.speaker_fragments_map[speaker]['words']]
            cleaned_words = [re.sub(r'[^\w\s-]', '', w.lower()) for w in words if w]
            unique_words_speaker = list(set(cleaned_words))
            level_counts = Counter(self.cefr_levels_dict.get(word, "Unknown") for word in unique_words_speaker)
            correctly_pronounced_complex_words = defaultdict(int)

            for word in unique_words_speaker:
                level = self.cefr_levels_dict.get(word, "Unknown")
                word_segments = [s for s in self.aligned_words_segments if s['speaker'] == speaker and re.sub(r'[^\w\s-]', '', s['text'].lower()) == word]
                if level in ["B2", "C1", "C2"] and word_segments:
                     if any(s.get('confidence', 0.0) >= self.confidence_threshold_for_correct for s in word_segments):
                         correctly_pronounced_complex_words[word] += len(word_segments)

            total_unique_words = sum(level_counts.values()) or 1
            vocab_dist_percent = {level: round((count / total_unique_words) * 100, 2) for level, count in level_counts.items()}
            sorted_vocab_dist = {level: vocab_dist_percent.get(level, 0.0) for level in self.CEFR_LEVEL_ORDER}

            estimated_level = "A1"
            if sorted_vocab_dist.get('C1', 0) > 3 or sorted_vocab_dist.get('C2', 0) > 1: estimated_level = "C1"
            elif sorted_vocab_dist.get('B2', 0) > 5: estimated_level = "B2"
            elif sorted_vocab_dist.get('B1', 0) > 10: estimated_level = "B1"
            elif sorted_vocab_dist.get('A2', 0) > 10: estimated_level = "A2"

            speaker_vocabulary_data[speaker] = {'total_unique_words': total_unique_words, 'vocabulary_distribution': sorted_vocab_dist, 'estimated_vocabulary_level': estimated_level, 'correctly_pronounced_complex_words': dict(correctly_pronounced_complex_words)}
            self.student_pronunciation_metrics.setdefault(speaker, {}).update(speaker_vocabulary_data[speaker])
            self.logger.info(f"Speaker {speaker} Vocabulary: Total Unique Words={total_unique_words}, Estimated Level={estimated_level}")

        # Plot 10: Vocabulary Distribution
        vocab_df = pd.DataFrame([{'Speaker': s, 'Level': l, 'Percentage': p} for s, d in speaker_vocabulary_data.items() for l, p in d['vocabulary_distribution'].items()])
        if not vocab_df.empty:
            vocab_df['Level'] = pd.Categorical(vocab_df['Level'], categories=self.CEFR_LEVEL_ORDER, ordered=True); plt.figure(figsize=(12, 7)); palette = sns.color_palette("viridis", n_colors=len(self.CEFR_LEVEL_ORDER)-1); palette.append((0.7, 0.7, 0.7)); sns.barplot(data=vocab_df, x='Speaker', y='Percentage', hue='Level', palette=dict(zip(self.CEFR_LEVEL_ORDER, palette)), dodge=True); plt.title('Vocabulary Distribution (% of Unique Words) per Speaker (via Gemini)'); plt.ylim(0, 100); plt.legend(title='CEFR Level', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(); plot_path = os.path.join(self.plots_dir, "plot_10_vocabulary_distribution_gemini.png"); plt.savefig(plot_path); plt.close(); self.logger.info(f"Plot saved: {plot_path}")

    # def _generate_reports(self):
    #     """Generates final JSON reports for each speaker (Step 5)."""
    #     self.logger.info("\n--- Step 5: Report Generation ---")
    #     for speaker in self.speaker_utterances_map.keys():
    #         metrics, speaking_time = self.student_pronunciation_metrics.get(speaker, {}), self.speaker_speaking_time_s_map.get(speaker, 0)
    #         speaker_mispronounced_df = self.df_mispronounced[self.df_mispronounced['speaker'] == speaker].copy() if not self.df_mispronounced.empty else pd.DataFrame()
    #         mispronounced_list_report = []
    #         if not speaker_mispronounced_df.empty:
    #             for _, row in speaker_mispronounced_df.iterrows():
    #                 #  relative_path = os.path.relpath(row['audio_clip_path_utterance'], self.base_output_dir) if row['audio_clip_path_utterance'] not in ["N/A", "ERROR_EXPORTING"] and os.path.exists(row['audio_clip_path_utterance']) else row['audio_clip_path_utterance']
    #                  mispronounced_list_report.append({"word": row['expected_word'], "start_time": round(row['start_ms_in_full_audio']/1000.0, 2), "expected_pronunciation": row['standard_pronunciation_ipa'], "pronunciation_score_proxy": row['word_pronunciation_score_proxy'], "complexity_level": row['lexical_complexity']})
            
    #         recommendations, overall_score, wpm, num_long_pauses, disfluency_count, num_spoken_words = [], metrics.get('overall_pronunciation_score_heuristic', 0), metrics.get('wpm', 0), metrics.get('num_long_pauses_between_utterances', 0), metrics.get('disfluency_count', 0), len(self.speaker_fragments_map.get(speaker, {}).get('words', []))
            
    #         if speaking_time > 15:
    #             recommendations.append("--- Fluency Feedback ---")
    #             if overall_score < 70: recommendations.append("Focus on speaking smoothly.")
    #             if wpm < 120 and speaking_time > 30: recommendations.append(f"- Your speaking speed is around {round(wpm,1)} WPM. Aiming for a slightly faster pace (e.g., 120-160 WPM) can sound more natural.")
    #             if num_long_pauses > 2 and speaking_time > 30: recommendations.append(f"- You had {num_long_pauses} pauses longer than {self.long_silence_threshold_ms/1000} seconds. Try to fill these pauses or transition more quickly.")
    #             if disfluency_count > (num_spoken_words * 0.05) and num_spoken_words > 50: recommendations.append(f"- We detected {disfluency_count} possible disfluencies or repetitions (like 'um' or repeating words). Try to plan your sentences slightly ahead.")

    #         recommendations.append("--- Pronunciation Feedback ---")
    #         if mispronounced_list_report:
    #             top_errors = sorted(mispronounced_list_report, key=lambda x: (x['pronunciation_score_proxy'], -self.CEFR_LEVEL_ORDER.index(x['complexity_level']) if x['complexity_level'] in self.CEFR_LEVEL_ORDER else float('-inf')))[:5]
    #             recommendations.append("Here are some words to practice:")
    #             for error in top_errors: recommendations.append(f"- Practice '{error['word']}' (level: {error['complexity_level']}). Listen to its standard pronunciation ({error['expected_pronunciation']}) and try to imitate it.")
    #         else:
    #             recommendations.append("Your pronunciation seems quite clear based on the analysis! Great job.")

    #         recommendations.append("--- Vocabulary Feedback ---")
    #         estimated_vocab_level = metrics.get('estimated_vocabulary_level', 'Unknown')
    #         recommendations.append(f"Your estimated active vocabulary level based on words used is: **{estimated_vocab_level}**.")
    #         vocab_dist = metrics.get('vocabulary_distribution', {})
    #         if vocab_dist.get('B2', 0) + vocab_dist.get('C1', 0) + vocab_dist.get('C2', 0) < 10: recommendations.append("- Consider learning and actively using more words at higher CEFR levels (B1, B2, C1).")
    #         else: recommendations.append("- You are already using some higher-level vocabulary! Continue expanding your range.")

    #         correct_complex_words = metrics.get('correctly_pronounced_complex_words', {})
    #         if correct_complex_words:
    #              recommendations.append("Positive note: You successfully used and pronounced these higher-level words:")
    #              for word, count in correct_complex_words.items(): recommendations.append(f"- '{word}' (used {count} time(s))")
    #         else:
    #              recommendations.append("Using more complex vocabulary will help you express nuanced ideas.")

    #         report = {"student_id": speaker, "lesson_id": self.transcript_data.get('id', 'unknown_lesson'), "total_speaking_time_s": round(speaking_time, 2), "fluency": {"words_per_minute": metrics.get('wpm', 0.0), "avg_pause_s_between_utterances": metrics.get('avg_pause_s_between_utterances', 0.0), "num_long_pauses_between_utterances": metrics.get('num_long_pauses_between_utterances', 0), "disfluency_count": metrics.get('disfluency_count', 0)}, "pronunciation_accuracy": {"overall_score_heuristic": metrics.get('overall_pronunciation_score_heuristic', 0), "asr_confidence_proxy": {"avg_confidence": round(1 - metrics.get('avg_phonetic_deviation_proxy', 1.0), 3) if isinstance(metrics.get('avg_phonetic_deviation_proxy'), float) else "N/A", "percent_words_above_threshold": metrics.get('percent_words_correct_proxy', 0.0)}, "mispronounced_words_details": mispronounced_list_report}, "vocabulary": {"estimated_vocabulary_level": metrics.get('estimated_vocabulary_level', 'Unknown'), "vocabulary_distribution_percent": metrics.get('vocabulary_distribution', {}), "total_unique_words": metrics.get('total_unique_words', 0), "correctly_pronounced_complex_words": metrics.get('correctly_pronounced_complex_words', {})}, "recommendations": recommendations, "notes": ["Pronunciation scores are proxies based on ASR confidence.", "Vocabulary levels are estimates based on a simplified word list."]}
    #         self.reports.append(report)
    #         report_path = os.path.join(self.reports_dir, f"report_{speaker}.json")
    #         with open(report_path, 'w', encoding='utf-8') as f:
    #             json.dump(report, f, indent=4, ensure_ascii=False)
    #         self.logger.info(f"Generated and saved report for {speaker} at {report_path}")

    #     if self.reports:
    #         self.logger.info("\n--- Example Generated Report (JSON Structure) for Speaker %s ---", self.reports[0]['student_id'])
    #         # Log a snippet of the json to avoid flooding the console
    #         self.logger.info(json.dumps(self.reports[0], indent=2)[:1000] + "\n...")
    
    def _generate_reports(self):
        """Generates final JSON reports for each speaker (Step 5)."""
        self.logger.info("\n--- Step 5: Report Generation ---")
        
        for speaker in self.speaker_utterances_map.keys():
            # --- Robustness Step 1: Use a default metrics dictionary ---
            # This guarantees that all keys exist, even if calculations failed for a speaker.
            default_metrics = {
                'wpm': 0.0,
                'avg_pause_s_between_utterances': 0.0,
                'num_long_pauses_between_utterances': 0,
                'disfluency_count': 0,
                'overall_pronunciation_score_heuristic': 0.0,
                'avg_phonetic_deviation_proxy': 1.0,
                'percent_words_correct_proxy': 0.0,
                'estimated_vocabulary_level': 'Unknown',
                'vocabulary_distribution': {},
                'total_unique_words': 0,
                'correctly_pronounced_complex_words': {}
            }
            # Update the defaults with any metrics that were successfully calculated for the speaker
            calculated_metrics = self.student_pronunciation_metrics.get(speaker, {})
            metrics = {**default_metrics, **calculated_metrics}

            speaking_time = self.speaker_speaking_time_s_map.get(speaker, 0)
            
            # Filter the mispronounced words DataFrame for the current speaker
            speaker_mispronounced_df = self.df_mispronounced[self.df_mispronounced['speaker'] == speaker].copy() if not self.df_mispronounced.empty else pd.DataFrame()
            
            mispronounced_list_report = []
            if not speaker_mispronounced_df.empty:
                for _, row in speaker_mispronounced_df.iterrows():
                    # Safely access all keys from the DataFrame row
                    mispronounced_list_report.append({
                        "expected_word": row.get('expected_word', 'n/a'),
                        "start_time": round(row.get('start_ms_in_full_audio', 0) / 1000.0, 2),
                        "expected_pronunciation": row.get('standard_pronunciation_ipa', 'n/a'),
                        "word_pronunciation_score_proxy": row.get('word_pronunciation_score_proxy', 0.0),
                        "complexity_level": row.get('lexical_complexity', 'Unknown'),
                        # Crucially include the audio path for the frontend
                        "word_audio_clip_path": row.get('word_audio_clip_path', 'N/A')
                    })
            
            # --- Your Original, Detailed Recommendation Logic ---
            recommendations = []
            overall_score = metrics.get('overall_pronunciation_score_heuristic', 0)
            wpm = metrics.get('wpm', 0)
            num_long_pauses = metrics.get('num_long_pauses_between_utterances', 0)
            disfluency_count = metrics.get('disfluency_count', 0)
            num_spoken_words = len(self.speaker_fragments_map.get(speaker, {}).get('words', []))
            
            if speaking_time > 15:
                recommendations.append("--- Fluency Feedback ---")
                if overall_score < 70: recommendations.append("Focus on speaking smoothly.")
                if wpm < 120 and speaking_time > 30: recommendations.append(f"- Your speaking speed is around {round(wpm,1)} WPM. Aiming for a slightly faster pace (e.g., 120-160 WPM) can sound more natural.")
                if num_long_pauses > 2 and speaking_time > 30: recommendations.append(f"- You had {num_long_pauses} pauses longer than {self.long_silence_threshold_ms/1000} seconds. Try to fill these pauses or transition more quickly.")
                if disfluency_count > (num_spoken_words * 0.05) and num_spoken_words > 50: recommendations.append(f"- We detected {disfluency_count} possible disfluencies or repetitions (like 'um' or repeating words). Try to plan your sentences slightly ahead.")

            recommendations.append("--- Pronunciation Feedback ---")
            if mispronounced_list_report:
                # Use a safe key access for sorting
                top_errors = sorted(mispronounced_list_report, key=lambda x: (x.get('word_pronunciation_score_proxy', 100), -self.CEFR_LEVEL_ORDER.index(x.get('complexity_level', 'Unknown')) if x.get('complexity_level') in self.CEFR_LEVEL_ORDER else float('-inf')))[:5]
                recommendations.append("Here are some words to practice:")
                for error in top_errors:
                    recommendations.append(f"- Practice '{error.get('expected_word', 'n/a')}' (level: {error.get('complexity_level', 'Unknown')}). Listen to its standard pronunciation ({error.get('expected_pronunciation', 'n/a')}) and try to imitate it.")
            else:
                recommendations.append("Your pronunciation seems quite clear based on the analysis! Great job.")

            recommendations.append("--- Vocabulary Feedback ---")
            estimated_vocab_level = metrics.get('estimated_vocabulary_level', 'Unknown')
            recommendations.append(f"Your estimated active vocabulary level based on words used is: **{estimated_vocab_level}**.")
            vocab_dist = metrics.get('vocabulary_distribution', {})
            if vocab_dist.get('B2', 0) + vocab_dist.get('C1', 0) + vocab_dist.get('C2', 0) < 10:
                recommendations.append("- Consider learning and actively using more words at higher CEFR levels (B1, B2, C1).")
            else:
                recommendations.append("- You are already using some higher-level vocabulary! Continue expanding your range.")

            correct_complex_words = metrics.get('correctly_pronounced_complex_words', {})
            if correct_complex_words:
                 recommendations.append("Positive note: You successfully used and pronounced these higher-level words:")
                 for word, count in correct_complex_words.items():
                     recommendations.append(f"- '{word}' (used {count} time(s))")
            else:
                 recommendations.append("Using more complex vocabulary will help you express nuanced ideas.")
            # --- End of Recommendation Logic ---

            # --- Final Report Assembly using the guaranteed 'metrics' dictionary ---
            report = {
              "student_id": speaker,
              "lesson_id": self.transcript_data.get('id', 'unknown_lesson'),
              "total_speaking_time_s": round(speaking_time, 2),
              "fluency": {
                  "words_per_minute": metrics['wpm'],
                  "avg_pause_s_between_utterances": metrics['avg_pause_s_between_utterances'],
                  "num_long_pauses_between_utterances": metrics['num_long_pauses_between_utterances'],
                  "disfluency_count": metrics['disfluency_count']
              },
              "pronunciation_accuracy": {
                  "overall_score_heuristic": metrics['overall_pronunciation_score_heuristic'],
                  "asr_confidence_proxy": {
                      "avg_confidence": round(1 - metrics['avg_phonetic_deviation_proxy'], 3),
                      "percent_words_above_threshold": metrics['percent_words_correct_proxy']
                  },
                  # This now correctly contains the audio path for each word
                  "mispronounced_words_details": mispronounced_list_report
              },
              "vocabulary": {
                  "estimated_vocabulary_level": metrics['estimated_vocabulary_level'],
                  "vocabulary_distribution_percent": metrics['vocabulary_distribution'],
                  "total_unique_words": metrics['total_unique_words'],
                  "correctly_pronounced_complex_words": metrics['correctly_pronounced_complex_words']
              },
              "recommendations": recommendations,
              "notes": [
                  "Pronunciation scores are proxies based on ASR confidence.",
                  "Vocabulary levels are estimates based on a simplified word list."
              ]
            }
            
            self.reports.append(report)
            report_path = os.path.join(self.reports_dir, f"report_{speaker}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Generated and saved report for {speaker} at {report_path}")

        if self.reports:
            self.logger.info("\n--- Example Generated Report (JSON Structure) for Speaker %s ---", self.reports[0]['student_id'])
            self.logger.info(json.dumps(self.reports[0], indent=2)[:1000] + "\n...")    

    def run_analysis(self):
        """
        Executes the full analysis pipeline from start to finish.
        """
        self.logger.info("--- Initializing Analysis ---")
        try:
            self._initialize_tools()
            self._load_data()
            self._prepare_data()
            self._generate_preliminary_plots()
            self._analyze_pronunciation()
            self._estimate_vocabulary_level() # Must be before error identification to fill CEFR levels
            self._identify_specific_errors()
            self._generate_reports()
            self.logger.info("--- ANALYSIS COMPLETE ---")
            self.logger.info(f"All outputs, logs, and reports are saved in: {self.base_output_dir}")
            return self.reports
        except Exception as e:
            self.logger.critical("An unrecoverable error occurred during the analysis pipeline.", exc_info=True)
            return None


if __name__ == '__main__':
    # -------------------- CONFIGURATION --------------------
    # Get API Key from environment variable for security
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        # Fallback for convenience, but replace with your key
        GEMINI_API_KEY = 'AIzaSyAgR91iOKvWnxaHXsF-vmXMX1TDrrpnx_o' # Replace with your actual key or set environment variable
        print("WARNING: Using hardcoded API Key. It is recommended to set the GEMINI_API_KEY environment variable.")

    # --- File Paths ---
    TRANSCRIPT_FILE = "../task-1/ground-truth/transcrip48.json"
    AUDIO_FILE = "../task-1/ground-truth/speaker_A_utt_48pppp.json"
    BASE_OUTPUT_DIR = "./classroom_audio_analysis-48"
    
    # -------------------- EXECUTION --------------------
    # Instantiate the analyzer framework
    analyzer = ClassroomAudioAnalyzer(
        transcript_file=TRANSCRIPT_FILE,
        audio_file=AUDIO_FILE,
        gemini_api_key=GEMINI_API_KEY,
        output_dir_base=BASE_OUTPUT_DIR
    )
    
    # Run the entire analysis
    final_reports = analyzer.run_analysis()

    if final_reports:
        print(f"\nSuccessfully completed analysis. {len(final_reports)} reports were generated.")
    else:
        print("\nAnalysis finished with errors. Please check the log file for details:")
        print(os.path.abspath(analyzer.log_file_path))