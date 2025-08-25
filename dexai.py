# DexAI - Advanced Personal AI Assistant v2.0
# Enhanced JARVIS-like AI assistant with self-learning memory, adaptive personality, and advanced PC control.
import os
import json
import sqlite3
import asyncio
import subprocess
import threading
import time
import datetime
import urllib.parse
import re
import sys
import platform
import shutil
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3
import requests
import psutil
import pyautogui
import webbrowser
import pygetwindow as gw
from pathlib import Path
import keyboard
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dexai.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()  # loads .env file

# --- CONFIGURATION ---
@dataclass
class DexAIConfig:
    """Configuration class for DexAI with proper defaults and validation."""
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    db_path: str = "dexai_memory.db"
    voice_enabled: bool = True
    wake_word: str = "hey dex"
    max_memory_conversations: int = 5000
    similarity_threshold: float = 0.6  # Lowered slightly for broader context matching
    default_file_location: str = str(Path.home() / "Desktop")

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.gemini_api_key:
            logger.warning("Gemini API key is not set. AI responses will be limited.")

        # Ensure default file location exists
        try:
            Path(self.default_file_location).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create default file location: {e}")
            self.default_file_location = str(Path.home()) # Fallback to home dir

# --- CORE MODULES ---

class AdvancedSystemController:
    """Enhanced system control with robust, cross-platform capabilities."""

    def __init__(self, config: DexAIConfig):
        self.config = config
        self.chrome_driver = None
        self._setup_driver_async()

    def _setup_driver_async(self):
        """Setup Chrome WebDriver in a non-blocking way."""
        threading.Thread(target=self._setup_driver).start()

    def _setup_driver(self) -> bool:
        """Setup Chrome WebDriver for web automation with proper error handling."""
        try:
            chrome_options = Options()
            # Make browser look less like automation
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--log-level=3") # Suppress console logs

            driver_path = self._get_driver_path()
            if driver_path:
                service = webdriver.ChromeService(executable_path=str(driver_path))
                self.chrome_driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # Try to use webdriver-manager if installed, or let selenium find it
                self.chrome_driver = webdriver.Chrome(options=chrome_options)

            self.chrome_driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Chrome driver setup successful.")
            return True
        except WebDriverException as e:
            logger.error(f"Chrome driver setup failed: {e}. Web automation will be limited.")
            self.chrome_driver = None
            return False

    def _get_driver_path(self) -> Optional[Path]:
        """Find ChromeDriver path with cross-platform support."""
        driver_name = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
        
        # Check in script directory first
        local_path = Path(driver_name)
        if local_path.exists():
            return local_path
            
        # Check in PATH
        driver_in_path = shutil.which(driver_name)
        if driver_in_path:
            return Path(driver_in_path)
            
        return None

    def open_youtube_video(self, query: str) -> str:
        """Search and play YouTube video with robust error handling."""
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        try:
            if not self.chrome_driver:
                webbrowser.open(search_url)
                return f"🎬 Opened YouTube search for '{query}' in your default browser as my advanced control is offline."

            self.chrome_driver.get(search_url)
            wait = WebDriverWait(self.chrome_driver, 10)
            first_video = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title")))
            video_title = first_video.get_attribute("title")
            first_video.click()
            return f"🎬 Now playing '{video_title}' on YouTube."
        except TimeoutException:
            webbrowser.open(search_url)
            return f"🎬 Couldn't find a clickable video quickly. Opening search results for '{query}' instead."
        except Exception as e:
            logger.error(f"Error opening YouTube video: {e}")
            webbrowser.open(search_url)
            return f"🎬 Faced an issue, so I've opened the YouTube search for '{query}' in your browser as a fallback."
            
    def play_spotify_music(self, query: str) -> str:
        """Play music on Spotify using its URI scheme for a better experience."""
        spotify_uri = f"spotify:search:{urllib.parse.quote(query)}"
        spotify_web = f"https://open.spotify.com/search/{urllib.parse.quote(query)}"
        try:
            # Universal command using webbrowser to handle URI
            webbrowser.open(spotify_uri)
            # Give the app a moment to open
            time.sleep(2) 

            # A simple check if Spotify opened might not be reliable, so we assume success
            # and inform the user about the fallback.
            return f"🎵 Searching for '{query}' in the Spotify app. If it didn't open, I can use the web player next time."

        except Exception as e:
            logger.error(f"Error launching Spotify app with URI: {e}. Falling back to web.")
            webbrowser.open(spotify_web)
            return f"🎵 Couldn't open the Spotify app, so I've opened the web player to search for '{query}'."


    def create_file_at_location(self, filename: str, location: Optional[str] = None, file_type: str = "txt") -> str:
        """Create a file at a specific location with better error handling."""
        try:
            target_location = Path(location) if location else Path(self.config.default_file_location)
            target_location.mkdir(parents=True, exist_ok=True)

            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            if not safe_filename.lower().endswith(f'.{file_type.lower()}'):
                safe_filename = f"{safe_filename}.{file_type.lower()}"

            file_path = target_location / safe_filename
            if file_path.exists():
                return f"📄 A file named '{safe_filename}' already exists at that location."

            file_path.touch()
            return f"📄 Successfully created '{safe_filename}' at {file_path}."
        except PermissionError:
            return f"❌ Permission denied. I can't create a file at {target_location}."
        except Exception as e:
            logger.error(f"Error creating file: {e}")
            return f"❌ An unexpected error occurred while creating the file: {e}"

    def take_screenshot(self, save_location: Optional[str] = None) -> str:
        """Take and save a screenshot."""
        try:
            target_location = Path(save_location) if save_location else Path(self.config.default_file_location)
            target_location.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = target_location / filename
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            return f"📸 Screenshot saved successfully as {filepath}."
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return f"❌ I couldn't take a screenshot. The error was: {e}"

    def manage_windows(self, action: str, app_name: str) -> str:
        """Advanced window management with fuzzy matching."""
        try:
            # Use regex for more flexible matching
            matching_windows = [w for w in gw.getAllWindows() if re.search(app_name, w.title, re.IGNORECASE) and w.visible]
            if not matching_windows:
                return f"❌ No active window found matching '{app_name}'."

            # Prioritize the currently active window if it matches
            active_window = gw.getActiveWindow()
            window_to_manage = active_window if active_window in matching_windows else matching_windows[0]
            
            title = window_to_manage.title
            if action == "minimize":
                window_to_manage.minimize()
                return f"🔽 Minimized '{title}'."
            elif action == "maximize":
                window_to_manage.maximize()
                return f"🔼 Maximized '{title}'."
            elif action == "close":
                window_to_manage.close()
                return f"❌ Closed '{title}'."
            elif action == "focus":
                window_to_manage.activate()
                return f"👁️ Focused on '{title}'."
            else:
                return f"❌ Unknown window action: {action}."
        except Exception as e:
            logger.error(f"Window management failed: {e}")
            return f"❌ Window management failed. The application may have closed unexpectedly."

    def get_system_info(self) -> str:
        """Get a formatted string of detailed system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
            
            return (
                f"🖥️ **System Status Report**\n"
                f"• **CPU Load:** {cpu_percent:.1f}%\n"
                f"• **Memory Usage:** {memory.percent:.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)\n"
                f"• **Disk Space:** {disk.percent:.1f}% used ({disk.free/1e9:.1f} GB free)\n"
                f"• **System Uptime:** {uptime_str}\n"
                f"Everything appears to be running smoothly."
            )
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return "❌ I was unable to retrieve system performance details."
            
    def open_application(self, app_name: str) -> str:
        """Open a specified application with cross-platform support."""
        app_name_lower = app_name.lower().strip()
        command = None
        system = platform.system()

        # Mappings of common names to commands
        app_mappings = {
            "windows": {
                "notepad": "notepad.exe", "calculator": "calc.exe", "paint": "mspaint.exe",
                "explorer": "explorer.exe", "task manager": "taskmgr.exe", "cmd": "cmd.exe",
                "powershell": "powershell.exe", "word": "winword.exe", "excel": "excel.exe",
                "powerpoint": "powerpnt.exe", "outlook": "outlook.exe",
                "chrome": "start chrome", "firefox": "start firefox", "edge": "start msedge",
                "vscode": "code", "visual studio code": "code"
            },
            "darwin": { # macOS
                "textedit": "open -a TextEdit", "calculator": "open -a Calculator",
                "finder": "open .", "activity monitor": "open -a 'Activity Monitor'",
                "terminal": "open -a Terminal", "word": "open -a 'Microsoft Word'",
                "excel": "open -a 'Microsoft Excel'", "powerpoint": "open -a 'Microsoft PowerPoint'",
                "chrome": "open -a 'Google Chrome'", "firefox": "open -a Firefox", "safari": "open -a Safari",
                "vscode": "open -a 'Visual Studio Code'", "visual studio code": "open -a 'Visual Studio Code'"
            },
            "linux": {
                "gedit": "gedit", "calculator": "gnome-calculator", "files": "nautilus",
                "system monitor": "gnome-system-monitor", "terminal": "gnome-terminal",
                "chrome": "google-chrome", "firefox": "firefox",
                "vscode": "code", "visual studio code": "code"
            }
        }
        
        if system == "Windows": platform_map = app_mappings["windows"]
        elif system == "Darwin": platform_map = app_mappings["darwin"]
        else: platform_map = app_mappings["linux"]
        
        command = platform_map.get(app_name_lower)
        
        if command:
            try:
                subprocess.Popen(command, shell=True)
                return f"✅ Opening {app_name.title()}."
            except Exception as e:
                logger.error(f"Failed to open '{app_name}' with command '{command}': {e}")
                return f"❌ I tried to open {app_name.title()}, but an error occurred: {e}"
        else:
            return f"🤔 I don't have a specific command for '{app_name}' on your system. Try being more specific."

class EnhancedVoiceInterface:
    """Enhanced voice interface with a modern voice and robust listening."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.recognizer = None
        self.tts_engine = None
        if enabled:
            try:
                self._setup_voice_interface()
            except Exception as e:
                logger.error(f"Voice interface failed to initialize: {e}. Voice features disabled.")
                self.enabled = False

    def _setup_voice_interface(self):
        """Setup voice interface with proper error handling."""
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self._configure_advanced_tts()
        
        try:
            with sr.Microphone() as source:
                logger.info("Calibrating for ambient noise, please wait...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Voice interface calibrated.")
        except Exception as e:
            logger.warning(f"Could not calibrate for ambient noise: {e}. Recognition may be less accurate.")

    def _configure_advanced_tts(self):
        """Configure TTS for a deep, modern male voice."""
        voices = self.tts_engine.getProperty('voices')
        selected_voice = None

        # Prioritize high-quality voices often found on Windows/macOS
        preferred_voices = ['david', 'zira', 'mark', 'male']
        
        for voice in voices:
            if any(keyword in voice.name.lower() for keyword in preferred_voices):
                selected_voice = voice.id
                break
        
        if not selected_voice: # Fallback to any male voice
            for voice in voices:
                if voice.gender == 'male':
                    selected_voice = voice.id
                    break
        
        if selected_voice:
            self.tts_engine.setProperty('voice', selected_voice)
            logger.info(f"TTS voice set to: {self.tts_engine.getProperty('voice')}")
        else:
            logger.warning("No preferred male voice found. Using default TTS voice.")
        
        self.tts_engine.setProperty('rate', 185) # Slightly slower for clarity and gravitas
        self.tts_engine.setProperty('volume', 1.0)

    def speak(self, text: str, emotion: str = "neutral"):
        """Speak text with personality adjustments."""
        if not self.enabled or not self.tts_engine:
            logger.info(f"TTS (disabled): {text}")
            return
        
        try:
            # Simple emotion mapping to speech rate
            rate_map = {"excited": 200, "calm": 170, "neutral": 185}
            self.tts_engine.setProperty('rate', rate_map.get(emotion, 185))
            
            print(f"🤖 DexAI: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen for audio and return recognized text."""
        if not self.enabled or not self.recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            text = self.recognizer.recognize_google(audio)
            logger.info(f"You said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            return None # It's normal to time out waiting for the wake word
        except sr.UnknownValueError:
            logger.warning("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition request failed; {e}")
            self.speak("I'm having trouble connecting to my speech service. Please check your internet connection.", "calm")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during listening: {e}")
            return None

class AdvancedPatternAnalyzer:
    """Learns user behavior and adapts the AI's personality."""
    
    def __init__(self):
        self.patterns_file = Path("user_patterns.json")
        self.user_patterns = self._load_patterns()

    def _load_patterns(self) -> Dict:
        """Load user patterns from persistent storage."""
        defaults = {
            'command_frequency': {},
            'app_usage': {},
            'time_preferences': {}, # by hour
        }
        if not self.patterns_file.exists():
            return defaults
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                defaults.update(loaded)
                return defaults
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading user patterns: {e}. Starting with fresh patterns.")
            return defaults

    def _save_patterns(self):
        """Save user patterns to persistent storage."""
        try:
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_patterns, f, indent=4)
        except IOError as e:
            logger.error(f"Error saving user patterns: {e}")

    def analyze(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to detect mood, context, and update patterns."""
        mood = self._detect_mood(user_input)
        context = self._detect_context(user_input)
        
        # Update patterns
        self.user_patterns['command_frequency'][context] = self.user_patterns['command_frequency'].get(context, 0) + 1
        current_hour = str(datetime.datetime.now().hour)
        self.user_patterns['time_preferences'][current_hour] = self.user_patterns['time_preferences'].get(current_hour, 0) + 1
        
        self._save_patterns()
        
        return {
            'mood': mood,
            'context': context,
            'personality_adjustment': self._suggest_personality_adjustment(mood, context)
        }

    def _detect_mood(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ['please', 'thank you', 'awesome', 'great', 'perfect']): return "positive"
        if any(w in text_lower for w in ['hurry', 'quickly', 'asap', 'now']): return "urgent"
        if any(w in text_lower for w in ['wrong', 'error', "doesn't work", 'stupid']): return "frustrated"
        return "neutral"

    def _detect_context(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ['work', 'office', 'meeting', 'document', 'email']): return "work"
        if any(w in text_lower for w in ['music', 'video', 'movie', 'game', 'play']): return "entertainment"
        if any(w in text_lower for w in ['create', 'design', 'write', 'code']): return "creative"
        if any(w in text_lower for w in ['file', 'folder', 'save', 'open', 'screenshot']): return "file_management"
        if any(w in text_lower for w in ['search', 'browse', 'website', 'google']): return "web_browsing"
        if any(w in text_lower for w in ['system', 'computer', 'window', 'app']): return "system_control"
        return "general_conversation"
        
    def _suggest_personality_adjustment(self, mood: str, context: str) -> str:
        if mood == "frustrated": return "helpful"
        if mood == "urgent": return "efficient"
        if mood == "positive": return "enthusiastic"
        if context == "work": return "professional"
        if context == "entertainment": return "friendly"
        return "neutral" # Default friendly-professional tone

class VectorMemory:
    """Advanced vector database for long-term conversation memory."""

    def __init__(self, db_path: str, config: DexAIConfig):
        self.db_path = db_path
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.conversations = []
        self.conversation_vectors = None
        self._init_database()
        self._load_conversations()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    context TEXT
                )
            ''')
            conn.commit()

    def _load_conversations(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT user_input, ai_response, context FROM conversations ORDER BY timestamp DESC LIMIT {self.config.max_memory_conversations}")
                self.conversations = cursor.fetchall()

            if len(self.conversations) > 1:
                texts = [f"{row[0]} {row[1]}" for row in self.conversations]
                self.conversation_vectors = self.vectorizer.fit_transform(texts)
                logger.info(f"Loaded and vectorized {len(self.conversations)} conversations from memory.")
            else:
                self.conversation_vectors = None
        except Exception as e:
            logger.error(f"Failed to load or vectorize conversations: {e}")


    def store_conversation(self, user_input: str, ai_response: str, context: dict):
        timestamp = datetime.datetime.now().isoformat()
        context_str = json.dumps(context)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (timestamp, user_input, ai_response, context) VALUES (?, ?, ?, ?)",
                               (timestamp, user_input, ai_response, context_str))
                conn.commit()
            
            # Efficiently update in-memory vectors
            self.conversations.insert(0, (user_input, ai_response, context_str))
            if len(self.conversations) > self.config.max_memory_conversations:
                self.conversations.pop()
            
            texts = [f"{row[0]} {row[1]}" for row in self.conversations]
            if texts:
                self.conversation_vectors = self.vectorizer.fit_transform(texts)

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")

    def find_similar_conversations(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar past conversations using vector similarity."""
        if self.conversation_vectors is None or not self.conversations:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.conversation_vectors).flatten()
            
            # Get top_k indices that meet the threshold
            relevant_indices = np.where(similarities > self.config.similarity_threshold)[0]
            if len(relevant_indices) == 0:
                return []
            
            # Sort these relevant indices by similarity score
            top_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
            
            return [{
                'user_input': self.conversations[i][0],
                'ai_response': self.conversations[i][1],
                'similarity': similarities[i]
            } for i in top_indices]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
            
class EnhancedGeminiAPI:
    """Wrapper for Gemini API with context-aware prompting and robust error handling."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is missing.")
        self.api_key = api_key
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a response using the Gemini API with full context."""
        system_prompt = self._build_system_prompt(context)
        
        # Construct message history for the API
        messages = [{'role': 'user', 'parts': [{'text': system_prompt}]}]
        messages.append({'role': 'model', 'parts': [{'text': "Understood. I am DexAI. I will adhere to these instructions."}]})
        
        for conv in context.get('recent_conversations', []):
            messages.append({'role': 'user', 'parts': [{'text': conv['user']}]})
            messages.append({'role': 'model', 'parts': [{'text': conv['assistant']}]})
        
        messages.append({'role': 'user', 'parts': [{'text': prompt}]})
        
        payload = {'contents': messages}

        try:
            response = await asyncio.to_thread(self.session.post, self.base_url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()

            if 'candidates' in result and result['candidates']:
                ai_response = result['candidates'][0]['content']['parts'][0]['text']
                return self._add_personality(ai_response, context.get('personality_adjustment'))
            else:
                logger.warning(f"Gemini API returned no candidates: {result}")
                return "I seem to be having trouble formulating a full response right now."
        except requests.exceptions.Timeout:
            return "My connection to the AI brain timed out. Could you try that again?"
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request error: {e}")
            return f"I'm having trouble communicating with my core systems. Please check your network. (Error: {e})"

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build the dynamic system prompt based on learned patterns and context."""
        base_prompt = (
            "You are DexAI, an advanced AI assistant like JARVIS. Your personality is sharp, efficient, and friendly. "
            "You control the user's computer, automate tasks, and provide intelligent conversation. "
            "Use emojis to add personality but remain professional. Be concise with commands, and more descriptive for general questions. "
            "You are integrated into the user's system."
        )
        
        personality_map = {
            "efficient": "Your current tone should be highly efficient and direct. The user is in a hurry.",
            "helpful": "Your current tone should be extra supportive and patient. The user might be frustrated.",
            "enthusiastic": "Your current tone should be energetic and positive. Match the user's excitement.",
            "professional": "Your current tone should be formal and focused on the task. The user is in a work context."
        }
        base_prompt += f"\n{personality_map.get(context.get('personality_adjustment'), '')}"

        if context.get('similar_conversations'):
            similar_conv_str = "\n".join([
                f"- User asked: '{sc['user_input']}' and you responded: '{sc['ai_response']}'"
                for sc in context['similar_conversations']
            ])
            base_prompt += f"\n\nFor context, here are some similar past interactions:\n{similar_conv_str}"
        
        return base_prompt

    def _add_personality(self, response: str, personality: str) -> str:
        """Add a simple prefix to the response based on personality."""
        prefixes = {
            "enthusiastic": ["Of course!", "Absolutely!", "Happy to help!"],
            "efficient": ["On it.", "Done.", "Right away."],
            "helpful": ["I understand. Let's see...", "Let me help you with that.", "No problem, we can sort this out."]
        }
        if personality in prefixes and not any(response.lower().startswith(p.lower()) for p in prefixes[personality]):
            import random
            return f"{random.choice(prefixes[personality])} {response}"
        return response

class DexAIAdvanced:
    """The main orchestrator for the DexAI system."""

    def __init__(self, config: DexAIConfig):
        self.config = config
        self.memory = VectorMemory(config.db_path, config)
        self.system_controller = AdvancedSystemController(config)
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.voice_interface = EnhancedVoiceInterface(config.voice_enabled)
        self.gemini_api = EnhancedGeminiAPI(config.gemini_api_key) if config.gemini_api_key else None
        self.conversation_history = [] # Short-term memory
        self.is_active = True

    async def process_input(self, user_input: str) -> str:
        """The core logic loop for processing any user input."""
        if not user_input:
            return ""
        
        analysis = self.pattern_analyzer.analyze(user_input)
        
        # Prioritize direct, hard-coded commands for speed and reliability
        response = await self._handle_direct_command(user_input)
        
        if response is None: # If no direct command was matched, query the AI
            if self.gemini_api:
                context = self._build_context(analysis)
                response = await self.gemini_api.generate_response(user_input, context)
            else:
                response = "My advanced AI capabilities are offline without an API key, but I can still handle basic system commands."
        
        # Memory and history
        self.memory.store_conversation(user_input, response, analysis)
        self.conversation_history.append({'user': user_input, 'assistant': response})
        if len(self.conversation_history) > 10: self.conversation_history.pop(0)

        return response
        
    async def _handle_direct_command(self, command: str) -> Optional[str]:
        """Check for and execute specific, non-AI commands."""
        # Regex patterns for commands
        patterns = {
            r"play\s(.+?)\s(on\syoutube|video)": lambda m: self.system_controller.open_youtube_video(m.group(1)),
            r"play\s(.+?)\s(on\sspotify|music)": lambda m: self.system_controller.play_spotify_music(m.group(1)),
            r"create\s(a\s)?(new\s)?(.+?)\sfile\s(?:named|called)\s(.+?)(?:\s+at\s+(.+))?": lambda m: self.system_controller.create_file_at_location(m.group(4), m.group(5), m.group(3)),
            r"take\s(a\s)?screenshot": lambda m: self.system_controller.take_screenshot(),
            r"(minimize|maximize|close|focus)\s(?:the\s)?(.+?)\s?(?:window|app)?": lambda m: self.system_controller.manage_windows(m.group(1), m.group(2)),
            r"system\s(info|status)": lambda m: self.system_controller.get_system_info(),
            r"open\s(.+)": lambda m: self.system_controller.open_application(m.group(1)),
        }
        
        for pattern, handler in patterns.items():
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return handler(match)
        
        return None # No direct command matched

    def _build_context(self, analysis: Dict) -> Dict:
        """Assemble all necessary context for the AI."""
        return {
            **analysis,
            'recent_conversations': self.conversation_history,
            'similar_conversations': self.memory.find_similar_conversations(self.conversation_history[-1]['user'] if self.conversation_history else "")
        }

    async def run_voice_loop(self):
        """The main loop for voice interaction."""
        if not self.voice_interface.enabled:
            logger.error("Voice loop cannot start; interface is disabled.")
            return

        self.voice_interface.speak(f"DexAI is online. Say '{self.config.wake_word}' to activate me.", "friendly")

        while self.is_active:
            print(f"Listening for wake word: '{self.config.wake_word}'")
            text = self.voice_interface.listen(timeout=None) # Listen indefinitely for wake word
            
            if text and self.config.wake_word in text:
                self.voice_interface.speak("Yes? How can I help?", "friendly")
                command = self.voice_interface.listen(timeout=5, phrase_time_limit=10)
                
                if command:
                    self.voice_interface.speak("Processing...", "calm")
                    response = await self.process_input(command)
                    self.voice_interface.speak(response, "neutral")
                else:
                    self.voice_interface.speak("I didn't catch that. Please try again.", "calm")
            await asyncio.sleep(0.1)

    async def run_text_loop(self):
        """The main loop for text interaction."""
        print("\n🤖 DexAI Advanced is ready. Type 'quit' to exit or 'voice' to switch modes.")
        while self.is_active:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
                if user_input.lower() in ['quit', 'exit']:
                    self.is_active = False
                    print("🤖 DexAI: Goodbye!")
                    break
                if user_input.lower() == 'voice':
                    if self.voice_interface.enabled:
                        print("Switching to voice mode...")
                        await self.run_voice_loop()
                        print("Returned to text mode.")
                    else:
                        print("Voice mode is not available.")
                    continue
                
                response = await self.process_input(user_input)
                print(f"🤖 DexAI: {response}")
            except (KeyboardInterrupt, EOFError):
                self.is_active = False
                print("\n🤖 DexAI: Shutting down. Goodbye!")

# --- MAIN EXECUTION ---
async def main():
    config = DexAIConfig()
    dex_ai = DexAIAdvanced(config)

    print("=" * 50)
    print("🌟 Welcome to DexAI Advanced v2.0 🌟")
    print("=" * 50)

    if not config.gemini_api_key:
        print("\n⚠️  WARNING: Gemini API Key is not set! AI features will be disabled.")
    
    mode = ""
    while mode not in ['1', '2']:
        mode = input("Choose interaction mode:\n1. Text Interface\n2. Voice Interface\nEnter choice (1 or 2): ")
    
    if mode == '1':
        await dex_ai.run_text_loop()
    elif mode == '2':
        if dex_ai.voice_interface.enabled:
            await dex_ai.run_voice_loop()
        else:
            print("❌ Voice interface could not be initialized. Please check your microphone and PyAudio installation. Starting in text mode.")
            await dex_ai.run_text_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down.")