# DexAI - Advanced Personal AI Assistant v2.0
# Enhanced JARVIS-like AI assistant with advanced PC control, web automation, and natural conversation
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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
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
import pyperclip
from pathlib import Path
import keyboard
import mouse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
import logging
import platform
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dexai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()  # loads .env file

# Enhanced Configuration
@dataclass
class DexAIConfig:
    """Configuration class for DexAI with proper defaults and validation."""
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    db_path: str = "dexai_memory.db"
    voice_enabled: bool = True
    wake_word: str = "hey dex"
    max_memory_conversations: int = 5000
    similarity_threshold: float = 0.7
    response_personality: str = "advanced_jarvis"
    chrome_driver_path: str = ""
    default_file_location: str = str(Path.home() / "Desktop")
    music_service: str = "spotify"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.gemini_api_key:
            logger.warning("Gemini API key is not set. AI responses will be limited.")
        
        # Ensure default file location exists
        try:
            Path(self.default_file_location).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create default file location: {e}")
            self.default_file_location = str(Path.home() / "Desktop")

class AdvancedSystemController:
    """Enhanced system control with web automation capabilities"""
    
    def __init__(self, config: DexAIConfig):
        self.config = config
        self.running_processes = {}
        self.chrome_driver = None
        self.active_browsers = []
        self._setup_driver()
    
    def _setup_driver(self) -> bool:
        """Setup Chrome WebDriver for web automation with proper error handling."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Cross-platform driver path handling
            driver_path = self._get_driver_path()
            if driver_path:
                self.chrome_driver = webdriver.Chrome(executable_path=str(driver_path), options=chrome_options)
            else:
                # Try to use driver from PATH
                self.chrome_driver = webdriver.Chrome(options=chrome_options)
            
            # Hide webdriver property
            self.chrome_driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Chrome driver setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Chrome driver setup failed: {e}")
            self.chrome_driver = None
            return False
    
    def _get_driver_path(self) -> Optional[Path]:
        """Get the ChromeDriver path with cross-platform support."""
        if self.config.chrome_driver_path:
            driver_path = Path(self.config.chrome_driver_path)
            if driver_path.exists():
                return driver_path
        
        # Try to find chromedriver in common locations
        possible_paths = [
            Path("chromedriver"),
            Path("chromedriver.exe"),
            Path("/usr/bin/chromedriver"),
            Path("/usr/local/bin/chromedriver"),
            Path.home() / "chromedriver",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def open_youtube_video(self, query: str) -> str:
        """Search and play YouTube video with robust error handling."""
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        
        try:
            if not self.chrome_driver:
                if not self._setup_driver():
                    webbrowser.open(search_url)
                    return f"🎬 Opened YouTube search for '{query}' in default browser (ChromeDriver not available)."
            
            self.chrome_driver.get(search_url)
            wait = WebDriverWait(self.chrome_driver, 10)
            first_video = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title")))
            video_title = first_video.get_attribute("title")
            first_video.click()
            return f"🎬 Now playing: '{video_title}' on YouTube."
            
        except Exception as e:
            logger.error(f"Error opening YouTube video: {e}")
            webbrowser.open(search_url)
            return f"🎬 Opened YouTube search for '{query}' (fallback mode due to error: {str(e)})."
    
    def play_spotify_music(self, query: str) -> str:
        """Play music on Spotify with multiple fallbacks."""
        try:
            # Try to open Spotify app first
            spotify_uri = f"spotify:search:{urllib.parse.quote(query)}"
            spotify_web = f"https://open.spotify.com/search/{urllib.parse.quote(query)}"
            
            # Cross-platform application opening
            if platform.system() == "Windows":
                try:
                    subprocess.run(f'start {spotify_uri}', shell=True, check=True, timeout=5)
                    return f"🎵 Searching for '{query}' in Spotify app."
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    webbrowser.open(spotify_web)
                    return f"🎵 Opened Spotify web player searching for '{query}'."
            else:
                # For macOS and Linux
                try:
                    subprocess.run(['open', spotify_uri], check=True, timeout=5)
                    return f"🎵 Searching for '{query}' in Spotify app."
                except:
                    webbrowser.open(spotify_web)
                    return f"🎵 Opened Spotify web player searching for '{query}'."
                    
        except Exception as e:
            logger.error(f"Error with Spotify: {e}")
            return f"❌ Couldn't access Spotify: {str(e)}. Please ensure Spotify is installed or try again."
    
    def open_canva_project(self, project_type: str = "document") -> str:
        """Open Canva with specific project type."""
        canva_urls = {
            "document": "https://www.canva.com/create/documents/",
            "presentation": "https://www.canva.com/create/presentations/",
            "poster": "https://www.canva.com/create/posters/",
            "logo": "https://www.canva.com/create/logos/",
            "instagram": "https://www.canva.com/create/instagram-posts/",
            "flyer": "https://www.canva.com/create/flyers/",
            "video": "https://www.canva.com/videos/",
            "social media": "https://www.canva.com/social-media/"
        }
        
        url = canva_urls.get(project_type.lower(), "https://www.canva.com/create/")
        
        try:
            webbrowser.open(url)
            return f"🎨 Opened Canva for creating {project_type}."
        except Exception as e:
            logger.error(f"Error opening Canva: {e}")
            return f"❌ Couldn't open Canva: {str(e)}. Please check your internet connection."
    
    def create_file_at_location(self, filename: str, location: str = None, file_type: str = "txt") -> str:
        """Create file at specific location with proper error handling."""
        try:
            if location is None:
                location = self.config.default_file_location
            
            # Ensure directory exists
            loc_path = Path(location)
            loc_path.mkdir(parents=True, exist_ok=True)
            
            # Create file with appropriate extension
            if not filename.lower().endswith(f'.{file_type.lower()}'):
                filename = f"{filename}.{file_type.lower()}"
            
            file_path = loc_path / filename
            
            # Content templates for different file types
            content_templates = {
                "txt": f"# {filename}\nCreated by DexAI on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "py": f"# {filename}\n# Created by DexAI on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nprint('Hello from DexAI!')\n",
                "html": f"<!DOCTYPE html>\n<html>\n<head>\n<title>{filename}</title>\n</head>\n<body>\n<h1>Created by DexAI</h1>\n</body>\n</html>\n"
            }
            
            content = content_templates.get(file_type.lower(), "")
            
            if content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"📄 Created '{filename}' at {file_path}."
            else:
                # For unsupported file types, create an empty file
                file_path.touch()
                return f"📄 Created empty file '{filename}' at {file_path} (unsupported file type)."
                
        except PermissionError:
            return f"❌ Permission denied: Cannot create file at {location}. Please check your permissions."
        except Exception as e:
            logger.error(f"Error creating file: {e}")
            return f"❌ Couldn't create file: {str(e)}. Please check the path and permissions."
    
    def open_office_app(self, app: str, create_new: bool = True) -> str:
        """Open Microsoft Office applications with proper error handling."""
        office_executables = {
            "word": "winword.exe" if platform.system() == "Windows" else "open -a Microsoft Word",
            "excel": "excel.exe" if platform.system() == "Windows" else "open -a Microsoft Excel",
            "powerpoint": "powerpnt.exe" if platform.system() == "Windows" else "open -a Microsoft PowerPoint",
            "outlook": "outlook.exe" if platform.system() == "Windows" else "open -a Microsoft Outlook",
            "onenote": "onenote.exe" if platform.system() == "Windows" else "open -a Microsoft OneNote"
        }
        
        app_lower = app.lower()
        if app_lower in office_executables:
            executable = office_executables[app_lower]
            
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(executable, shell=True)
                else:
                    subprocess.Popen(executable.split(), shell=False)
                    
                return f"📝 Opened {app.capitalize()} {'with a new document' if create_new else ''}."
                
            except FileNotFoundError:
                return f"❌ {app.capitalize()} not found. Please ensure it is installed."
            except Exception as e:
                return f"❌ Couldn't open {app.capitalize()}: {str(e)}."
        else:
            return f"❌ Unknown Office application: {app}. I can open Word, Excel, PowerPoint, Outlook, or OneNote."
    
    def control_media(self, action: str) -> str:
        """Control media playback with robust error handling."""
        try:
            media_keys = {
                "play": "playpause",
                "pause": "playpause",
                "next": "nexttrack",
                "previous": "prevtrack",
                "stop": "stop",
                "volume_up": "volumeup",
                "volume_down": "volumedown",
                "mute": "volumemute"
            }
            
            action_lower = action.lower()
            if action_lower in media_keys:
                keyboard.press_and_release(media_keys[action_lower])
                return f"🎵 {action.capitalize()} media."
            else:
                return f"❌ Unknown media action: {action}. I can play, pause, skip, control volume, or mute."
                
        except Exception as e:
            logger.error(f"Media control failed: {e}")
            return f"❌ Media control failed: {str(e)}. Ensure media is playing and your system supports media keys."
    
    def take_screenshot(self, save_location: str = None) -> str:
        """Take and save screenshot with proper error handling."""
        try:
            if save_location is None:
                save_location = self.config.default_file_location
            
            loc_path = Path(save_location)
            loc_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = loc_path / filename
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            return f"📸 Screenshot saved as {filepath}."
            
        except PermissionError:
            return f"❌ Permission denied: Cannot save screenshot to {save_location}. Please check your permissions."
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return f"❌ Screenshot failed: {str(e)}. Please check permissions or disk space."
    
    def manage_windows(self, action: str, app_name: str = None) -> str:
        """Advanced window management with proper error handling."""
        try:
            if action == "list":
                windows = gw.getAllWindows()
                # Filter out empty titles, invisible windows, and specific system windows
                active_windows = [w.title for w in windows if w.title.strip() and w.visible and w.title not in ["Program Manager", "Default IME"]]
                
                if active_windows:
                    return f"🪟 Active windows: {', '.join(active_windows[:10])}{'...' if len(active_windows) > 10 else ''}."
                else:
                    return "🪟 No active windows found."
            
            if not app_name:
                return "❌ Please specify an application name for window management (e.g., 'minimize Chrome')."
            
            # Use regex for more flexible matching
            matching_windows = [w for w in gw.getAllWindows() if re.search(app_name, w.title, re.IGNORECASE)]
            if not matching_windows:
                return f"❌ No window found with title containing '{app_name}'."
            
            # Prioritize visible windows, then the first match
            window_to_manage = next((w for w in matching_windows if w.visible), matching_windows[0])
            
            actions = {
                "minimize": (window_to_manage.minimize, f"🔽 Minimized '{window_to_manage.title}'."),
                "maximize": (window_to_manage.maximize, f"🔼 Maximized '{window_to_manage.title}'."),
                "close": (window_to_manage.close, f"❌ Closed '{window_to_manage.title}'."),
                "focus": (window_to_manage.activate, f"👁️ Focused on '{window_to_manage.title}'.")
            }
            
            if action in actions:
                func, message = actions[action]
                func()
                return message
            else:
                return f"❌ Unknown window management action: {action}."
                
        except Exception as e:
            logger.error(f"Window management failed: {e}")
            return f"❌ Window management failed: {str(e)}. Ensure the application is running."
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information with proper error handling."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            boot_time_timestamp = psutil.boot_time()
            uptime_seconds = time.time() - boot_time_timestamp
            
            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'running_processes': len(psutil.pids()),
                'uptime': uptime_seconds
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def open_application(self, app_name: str) -> bool:
        """Open a specified application with cross-platform support."""
        app_name_lower = app_name.lower()
        
        # Common application mappings with cross-platform support
        app_mappings = {
            "notepad": "notepad.exe" if platform.system() == "Windows" else "open -a TextEdit",
            "calculator": "calc.exe" if platform.system() == "Windows" else "open -a Calculator",
            "paint": "mspaint.exe" if platform.system() == "Windows" else "open -a Paintbrush",
            "cmd": "cmd.exe" if platform.system() == "Windows" else "open -a Terminal",
            "powershell": "powershell.exe" if platform.system() == "Windows" else "open -a Terminal",
            "chrome": "chrome.exe" if platform.system() == "Windows" else "open -a Google Chrome",
            "firefox": "firefox.exe" if platform.system() == "Windows" else "open -a Firefox",
            "edge": "msedge.exe" if platform.system() == "Windows" else "open -a Microsoft Edge",
            "explorer": "explorer.exe" if platform.system() == "Windows" else "open",
            "settings": "ms-settings:" if platform.system() == "Windows" else "open -a System Preferences",
            "task manager": "taskmgr.exe" if platform.system() == "Windows" else "open -a Activity Monitor",
            "vlc": "vlc.exe" if platform.system() == "Windows" else "open -a VLC",
            "discord": "discord.exe" if platform.system() == "Windows" else "open -a Discord",
            "steam": "steam.exe" if platform.system() == "Windows" else "open -a Steam",
            "code": "code.exe" if platform.system() == "Windows" else "code" if shutil.which("code") else "open -a Visual Studio Code",
            "terminal": "wt.exe" if platform.system() == "Windows" else "open -a Terminal",
            "word": "winword.exe" if platform.system() == "Windows" else "open -a Microsoft Word",
            "excel": "excel.exe" if platform.system() == "Windows" else "open -a Microsoft Excel",
            "powerpoint": "powerpnt.exe" if platform.system() == "Windows" else "open -a Microsoft PowerPoint",
            "outlook": "outlook.exe" if platform.system() == "Windows" else "open -a Microsoft Outlook",
            "onenote": "onenote.exe" if platform.system() == "Windows" else "open -a Microsoft OneNote",
            "spotify": "spotify.exe" if platform.system() == "Windows" else "open -a Spotify",
            "photoshop": "photoshop.exe" if platform.system() == "Windows" else "open -a Photoshop",
            "illustrator": "illustrator.exe" if platform.system() == "Windows" else "open -a Illustrator"
        }
        
        executable = app_mappings.get(app_name_lower)
        if executable:
            try:
                if platform.system() == "Windows":
                    if executable.startswith("ms-settings:"):
                        subprocess.Popen(["start", executable], shell=True)
                    else:
                        subprocess.Popen(executable, shell=True)
                else:
                    # macOS and Linux
                    subprocess.Popen(executable.split(), shell=False)
                return True
            except Exception as e:
                logger.error(f"Error opening application '{app_name}': {e}")
                return False
        else:
            # Attempt to open directly if not in mappings
            try:
                subprocess.Popen(app_name.split(), shell=(platform.system() == "Windows"))
                return True
            except Exception as e:
                logger.error(f"Error opening application '{app_name}': {e}")
                return False

class EnhancedVoiceInterface:
    """Enhanced voice interface with better TTS and robust listening."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.recognizer = None
        self.tts_engine = None
        
        if enabled:
            self._setup_voice_interface()
    
    def _setup_voice_interface(self):
        """Setup voice interface with proper error handling."""
        try:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS engine
            self._configure_advanced_tts()
            
            # Calibrate for ambient noise
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.info("Voice interface calibrated for ambient noise.")
            except Exception as e:
                logger.warning(f"Could not calibrate for ambient noise: {e}")
                
        except Exception as e:
            logger.error(f"Voice interface setup failed: {e}")
            self.enabled = False
            self.recognizer = None
            self.tts_engine = None
    
    def _configure_advanced_tts(self):
        """Configure advanced text-to-speech with deep male voice."""
        try:
            voices = self.tts_engine.getProperty('voices')
            selected_voice = None
            
            # Look for preferred male voices
            preferred_voices_keywords = ['david', 'mark', 'paul', 'male', 'zira', 'microsoft zira']
            
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                if any(keyword in voice_name for keyword in preferred_voices_keywords) or \
                   any(keyword in voice_id for keyword in preferred_voices_keywords):
                    if 'male' in voice.name or any(male_name in voice_name for male_name in ['david', 'mark', 'paul']):
                        selected_voice = voice.id
                        break
            
            # Fallback to any male voice
            if not selected_voice:
                for voice in voices:
                    if 'male' in voice.name.lower():
                        selected_voice = voice.id
                        break
            
            # Set voice if found
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
            else:
                logger.warning("No preferred male voice found. Using default TTS voice.")
            
            # Voice settings
            self.tts_engine.setProperty('rate', 190)
            self.tts_engine.setProperty('volume', 0.9)
            
        except Exception as e:
            logger.error(f"TTS configuration error: {e}")
            self.enabled = False
            self.tts_engine = None
    
    def speak_with_personality(self, text: str, emotion: str = "neutral"):
        """Speak with different personality traits."""
        if not self.enabled or not self.tts_engine:
            logger.debug(f"TTS Disabled: {text}")
            return
        
        try:
            # Adjust speech rate based on emotion
            rate_adjustments = {
                "excited": 210,
                "calm": 170,
                "serious": 160,
                "friendly": 190,
                "neutral": 190
            }
            
            original_rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', rate_adjustments.get(emotion, 190))
            
            # Add natural pauses for longer texts
            if len(text) > 100:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        self.tts_engine.say(sentence)
                        self.tts_engine.runAndWait()
                        if i < len(sentences) - 1:
                            time.sleep(0.3)
            else:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            # Restore original rate
            self.tts_engine.setProperty('rate', original_rate)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
    
    def listen(self, timeout: Optional[int] = None, phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """Listen for audio input and return recognized text."""
        if not self.enabled or not self.recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            logger.info("Processing audio...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during listening: {e}")
            return None

class AdvancedPatternAnalyzer:
    """Enhanced pattern analysis with user behavior learning."""
    
    def __init__(self, memory):
        self.memory = memory
        self.user_patterns = {
            'common_requests': {},
            'time_preferences': {},
            'day_preferences': {},
            'topics_of_interest': {},
            'command_frequency': {},
            'app_usage': {},
            'creative_preferences': {},
            'work_patterns': {}
        }
        self.conversation_mood = "neutral"
        self.patterns_file = "user_patterns.json"
        self._load_user_patterns()
    
    def _load_user_patterns(self):
        """Load user patterns from persistent storage."""
        if Path(self.patterns_file).exists():
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    loaded_patterns = json.load(f)
                    for key, value in loaded_patterns.items():
                        if key in self.user_patterns and isinstance(self.user_patterns[key], dict) and isinstance(value, dict):
                            self.user_patterns[key].update(value)
                        else:
                            self.user_patterns[key] = value
                logger.info("User patterns loaded successfully.")
            except json.JSONDecodeError as e:
                logger.error(f"User patterns file is corrupted: {e}")
            except Exception as e:
                logger.error(f"Error loading user patterns: {e}")
        else:
            logger.info("No existing user patterns file found. Starting fresh.")
    
    def _save_user_patterns(self):
        """Save user patterns to persistent storage."""
        try:
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_patterns, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving user patterns: {e}")
    
    def _get_enhanced_preferences(self) -> Dict[str, Any]:
        """Aggregate and return key user preferences."""
        preferences = {}
        
        if self.user_patterns['app_usage']:
            preferences['favorite_apps'] = sorted(
                self.user_patterns['app_usage'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        
        if self.user_patterns['time_preferences']:
            valid_hours = {int(k): v for k, v in self.user_patterns['time_preferences'].items() if str(k).isdigit()}
            if valid_hours:
                most_active_hour = int(max(valid_hours, key=valid_hours.get))
                preferences['active_hour'] = f"{most_active_hour}:00 - {most_active_hour+1}:00"
        
        if self.user_patterns['day_preferences']:
            valid_days = {int(k): v for k, v in self.user_patterns['day_preferences'].items() if str(k).isdigit()}
            if valid_days:
                days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                most_active_day_index = int(max(valid_days, key=valid_days.get))
                preferences['active_day'] = days_of_week[most_active_day_index]
        
        if self.user_patterns['command_frequency']:
            most_common_command = max(self.user_patterns['command_frequency'], key=self.user_patterns['command_frequency'].get)
            preferences['common_command_type'] = most_common_command
        
        return preferences
    
    def analyze_advanced_patterns(self, user_input: str, response_generated: str) -> Dict[str, Any]:
        """Enhanced pattern analysis based on user input and AI response."""
        mood = self._detect_mood(user_input)
        context = self._detect_context(user_input)
        command_type = self._classify_advanced_command(user_input)
        
        # Update patterns
        self.user_patterns['command_frequency'][str(command_type)] = self.user_patterns['command_frequency'].get(str(command_type), 0) + 1
        
        detected_app = self._extract_app_name(user_input)
        if detected_app != 'unknown':
            self.user_patterns['app_usage'][str(detected_app)] = self.user_patterns['app_usage'].get(str(detected_app), 0) + 1
        
        # Time-based learning
        current_hour = datetime.datetime.now().hour
        self.user_patterns['time_preferences'][str(current_hour)] = self.user_patterns['time_preferences'].get(str(current_hour), 0) + 1
        
        # Day-based learning
        current_day = datetime.datetime.now().weekday()
        self.user_patterns['day_preferences'][str(current_day)] = self.user_patterns['day_preferences'].get(str(current_day), 0) + 1
        
        # Update conversation mood
        if mood == "positive":
            self.conversation_mood = "positive"
        elif mood == "frustrated":
            self.conversation_mood = "frustrated"
        elif mood == "urgent":
            self.conversation_mood = "urgent"
        elif mood in ["casual", "neutral"] and self.conversation_mood not in ["positive", "frustrated", "urgent"]:
            self.conversation_mood = mood
        
        self._save_user_patterns()
        
        return {
            'command_type': command_type,
            'mood': mood,
            'context': context,
            'user_preferences': self._get_enhanced_preferences(),
            'suggestions': self._generate_smart_suggestions(user_input),
            'personality_adjustment': self._suggest_personality_adjustment(mood, context)
        }
    
    def _detect_mood(self, text: str) -> str:
        """Detect user mood from input using keyword matching."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['please', 'thank', 'awesome', 'great', 'love', 'good', 'happy', 'excellent', 'perfect', 'nice']):
            return "positive"
        elif any(word in text_lower for word in ['urgent', 'quickly', 'asap', 'now', 'fast', 'immediate', 'hurry']):
            return "urgent"
        elif any(word in text_lower for word in ['help', 'stuck', 'problem', 'issue', 'wrong', 'error', 'frustrated', 'can\'t', 'bug', 'trouble']):
            return "frustrated"
        elif any(word in text_lower for word in ['hey', 'hi', 'hello', 'what\'s up', 'casual', 'just', 'wondering', 'tell me']):
            return "casual"
        else:
            return "neutral"
    
    def _detect_context(self, text: str) -> str:
        """Detect conversation context using keyword matching."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['work', 'office', 'meeting', 'presentation', 'email', 'report', 'project', 'task', 'deadline']):
            return "work"
        elif any(word in text_lower for word in ['music', 'video', 'movie', 'game', 'fun', 'entertainment', 'play', 'watch', 'listen', 'show']):
            return "entertainment"
        elif any(word in text_lower for word in ['create', 'make', 'design', 'build', 'write', 'art', 'develop', 'code', 'draw', 'edit', 'photoshop', 'illustrator', 'canva']):
            return "creative"
        elif any(word in text_lower for word in ['file', 'folder', 'document', 'save', 'open', 'delete', 'organize', 'directory', 'screenshot', 'capture']):
            return "file_management"
        elif any(word in text_lower for word in ['search', 'browse', 'website', 'google', 'internet', 'webpage', 'online', 'url', 'site']):
            return "web_browsing"
        elif any(word in text_lower for word in ['system', 'computer', 'pc', 'windows', 'settings', 'control', 'app', 'program', 'task manager', 'info', 'status', 'window']):
            return "system_control"
        elif any(word in text_lower for word in ['weather', 'news', 'time', 'date', 'fact', 'information', 'define', 'what is', 'how to']):
            return "information_retrieval"
        else:
            return "general"
    
    def _classify_advanced_command(self, text: str) -> str:
        """Enhanced command classification based on keywords."""
        text_lower = text.lower()
        command_patterns = {
            'media_control': ['play', 'pause', 'music', 'video', 'youtube', 'spotify', 'next track', 'volume', 'song', 'audio', 'stop media'],
            'office_work': ['word', 'excel', 'powerpoint', 'document', 'spreadsheet', 'presentation', 'outlook', 'onenote'],
            'creative_work': ['canva', 'design', 'create design', 'make art', 'draw', 'edit photo', 'photoshop', 'illustrator'],
            'file_operations': ['file', 'folder', 'create file', 'save file', 'open file', 'delete file', 'screenshot', 'capture screen'],
            'system_control': ['window', 'minimize', 'maximize', 'close app', 'focus', 'system info', 'task manager', 'open app', 'launch app', 'start app', 'settings'],
            'web_browsing': ['search', 'browse', 'website', 'google', 'open browser', 'go to', 'internet'],
            'communication': ['email', 'message', 'send', 'call', 'skype', 'teams', 'discord'],
            'information_retrieval': ['what is', 'tell me about', 'how to', 'define', 'weather', 'news', 'time', 'date', 'fact']
        }
        
        for cmd_type, keywords in command_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return cmd_type
        return 'conversation'
    
    def _extract_app_name(self, text: str) -> str:
        """Extract application name from text."""
        apps = ['youtube', 'spotify', 'excel', 'word', 'canva', 'powerpoint', 'outlook', 'onenote',
                'chrome', 'firefox', 'edge', 'notepad', 'calculator', 'paint', 'cmd', 'powershell',
                'discord', 'steam', 'vlc', 'code', 'terminal', 'photoshop', 'illustrator']
        
        for app in apps:
            if app in text.lower():
                return app
        return 'unknown'
    
    def _generate_smart_suggestions(self, current_input: str) -> List[str]:
        """Generate intelligent suggestions based on current input and user patterns."""
        suggestions = []
        
        if 'music' in current_input.lower() or 'spotify' in current_input.lower():
            suggestions.extend([
                "I can also control playback with commands like 'pause', 'next track', or 'volume up'.",
                "Would you like me to create a playlist for you on Spotify?",
                "I can search for music on YouTube as well."
            ])
        elif 'file' in current_input.lower() or 'document' in current_input.lower():
            suggestions.extend([
                "I can create files in different formats - just specify .txt, .py, .html, etc.",
                "Need me to organize your files or take a screenshot?",
                "I can also open existing files if you tell me the path."
            ])
        elif any(app in current_input.lower() for app in ['word', 'excel', 'powerpoint', 'office']):
            suggestions.extend([
                "I can help you with other Office tasks too - just ask!",
                "Would you like me to create a template or open an existing file?",
                "I can also open Outlook for your emails."
            ])
        elif 'web' in current_input.lower() or 'browser' in current_input.lower() or 'search' in current_input.lower():
            suggestions.extend([
                "What website would you like to visit?",
                "I can search for information on Google for you.",
                "Do you want to open a new tab or window?"
            ])
        
        # Personality-based suggestions
        if self.user_patterns.get('creative_preferences') and self.user_patterns['creative_preferences'].get('design_count', 0) > 5:
            suggestions.append("It seems you enjoy creative tasks! Can I help you with a new design in Canva?")
        
        # Time-based suggestions
        current_hour = datetime.datetime.now().hour
        if self.user_patterns.get('time_preferences') and self.user_patterns['time_preferences'].get(str(current_hour), 0) > 5 and current_hour >= 20:
            suggestions.append("It's getting late. Remember to take a break! Can I play some relaxing music for you?")
        
        return list(dict.fromkeys(suggestions))[:3]
    
    def _suggest_personality_adjustment(self, mood: str, context: str) -> str:
        """Suggest personality adjustments based on detected mood and context."""
        adjustments = {
            ("urgent", "work"): "efficient",
            ("frustrated", "any"): "helpful",
            ("positive", "entertainment"): "enthusiastic",
            ("casual", "any"): "friendly",
            ("neutral", "creative"): "encouraging",
            ("neutral", "work"): "professional",
            ("positive", "general"): "friendly"
        }
        
        return adjustments.get((mood, context), adjustments.get((mood, "any"), "neutral"))

class EnhancedGeminiAPI:
    """Enhanced Gemini API with better prompting and error handling."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("❌ Gemini API key is not set. Please provide a valid API key.")
        
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        self.conversation_history = []
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    async def generate_enhanced_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate enhanced response with personality and context."""
        try:
            system_prompt = self._build_system_prompt(context)
            
            messages = []
            
            # Add recent conversation history
            for conv in context.get('recent_conversations', []):
                messages.append({'role': 'user', 'parts': [{'text': conv['user']}]})
                messages.append({'role': 'model', 'parts': [{'text': conv['assistant']}]})
            
            # Add current prompt
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            messages.append({'role': 'user', 'parts': [{'text': full_prompt}]})
            
            data = {
                'contents': messages,
                'generationConfig': {
                    'temperature': 0.8,
                    'topP': 0.9,
                    'topK': 40,
                    'maxOutputTokens': 1024,
                },
                'safetySettings': [
                    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
                    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
                    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
                    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
                ]
            }
            
            # Make the request
            response = self.session.post(
                f"{self.base_url}?key={self.api_key}",
                json=data,
                timeout=45
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                ai_response = result['candidates'][0]['content']['parts'][0]['text']
                return self._enhance_response_personality(ai_response, context)
            else:
                logger.warning(f"Gemini API returned no candidates: {result}")
                return "I'm sorry, I couldn't generate a response at this moment. The AI model didn't provide a valid output."
                
        except requests.exceptions.Timeout:
            return "I'm taking a bit longer to think. Please try again in a moment, or rephrase your request."
        except requests.exceptions.ConnectionError:
            return "It seems I'm having trouble connecting to the internet. Please check your connection and try again."
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Gemini API HTTP error: {http_err} - {http_err.response.text}")
            return f"I'm having some technical difficulties right now. Could you try that again? (Error: {http_err.response.status_code})"
        except Exception as e:
            logger.error(f"Unexpected error in Gemini API call: {e}")
            return f"I encountered an issue: {str(e)}. Let me try to help you in a different way."
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build enhanced system prompt for Gemini API."""
        mood = context.get('mood', 'neutral')
        context_type = context.get('context', 'general')
        personality = context.get('personality_adjustment', 'friendly')
        
        base_prompt = """You are DexAI, an advanced AI assistant similar to JARVIS from Iron Man. You are:
- Highly intelligent and capable
- Friendly, helpful, and personable
- Able to control computer systems and applications
- Knowledgeable about technology and productivity
- Conversational and engaging, not robotic
- Always provide concise and direct answers when a specific action is requested.
- When asked a general question, provide a helpful and informative response.
- Use emojis appropriately to convey tone and enhance readability.
"""
        
        personality_adjustments = {
            'efficient': "Keep responses concise and action-oriented. The user seems to be in a hurry. Focus on direct answers.",
            'helpful': "Be extra supportive and offer multiple solutions. The user may be frustrated. Show empathy.",
            'enthusiastic': "Be more energetic and excited. Match the user's positive mood. Use positive language.",
            'encouraging': "Be supportive and motivating. Help boost the user's confidence in their creative work. Offer creative ideas.",
            'friendly': "Be warm and conversational. Keep the tone light and personable. Use common greetings.",
            'professional': "Maintain a formal and respectful tone. Provide precise and factual information. Avoid slang or overly casual language."
        }
        
        if personality in personality_adjustments:
            base_prompt += f"\nCurrent interaction style: {personality_adjustments[personality]}"
        
        if context.get('user_preferences'):
            base_prompt += f"\nUser patterns and preferences: {context['user_preferences']}. Tailor your responses to these preferences."
        
        if context.get('similar_conversations'):
            similar_conv_str = "\n".join([
                f"Past User: {sc['user_input']}\nPast DexAI: {sc['ai_response']}"
                for sc in context['similar_conversations']
            ])
            base_prompt += f"\nRelevant past conversations for context:\n{similar_conv_str}"
        
        base_prompt += "\nYour goal is to be the most advanced and helpful personal AI assistant."
        return base_prompt
    
    def _enhance_response_personality(self, response: str, context: Dict[str, Any]) -> str:
        """Add personality touches and emojis to the AI's response."""
        mood = context.get('mood', 'neutral')
        personality = context.get('personality_adjustment', 'friendly')
        
        personality_prefixes = {
            'positive': ['Absolutely!', 'I\'d be delighted to help!', 'Perfect!', 'Fantastic!'],
            'urgent': ['Right away!', 'On it immediately!', 'Let me handle that quickly!', 'Understood, processing now.'],
            'frustrated': ['I understand, let me help you with that.', 'No worries, I\'ve got you covered.', 'Let\'s get this sorted.'],
            'casual': ['Hey there!', 'Sure thing!', 'You got it!', 'No problem!'],
            'efficient': ['Affirmative.', 'Task initiated.', 'Proceeding as requested.'],
            'enthusiastic': ['Awesome!', 'Let\'s do this!', 'Excited to help!'],
            'encouraging': ['You got this!', 'Keep up the great work!', 'I believe in your creativity!'],
            'professional': ['Understood.', 'Processing your request.', 'Certainly.']
        }
        
        prefix_list = personality_prefixes.get(personality, personality_prefixes.get(mood, []))
        if prefix_list and not response.startswith(tuple(prefix_list)):
            import random
            prefix = random.choice(prefix_list)
            response = f"{prefix} {response}"
        
        # Add emojis for visual enhancement
        if len(response) > 20 and not any(char in response for char in ['❌', '🚫', 'Error', 'Couldn\'t', 'failed', 'issue']):
            import random
            if random.random() > 0.6:
                emojis = {
                    'positive': ['✨', '🚀', '💫', '✅', '👍'],
                    'casual': ['😊', '👍', '🎯', '👋'],
                    'enthusiastic': ['🎉', '🌟', '🚀', '💡'],
                    'friendly': ['😊', '👋', '👍'],
                    'efficient': ['⚙️', '✔️'],
                    'creative': ['🎨', '💡', '✨']
                }
                emoji_choice = emojis.get(personality, emojis.get(mood, []))
                if emoji_choice:
                    emoji = random.choice(emoji_choice)
                    response = f"{emoji} {response}"
        
        return response

class DexAIAdvanced:
    """Advanced DexAI with enhanced capabilities and robust interaction."""
    
    def __init__(self, config: DexAIConfig):
        self.config = config
        self.memory = VectorMemory(config.db_path, config)
        self.system_controller = AdvancedSystemController(config)
        self.gemini_api = None
        
        # Initialize Gemini API only if API key is provided
        if config.gemini_api_key:
            try:
                self.gemini_api = EnhancedGeminiAPI(config.gemini_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
        
        self.voice_interface = EnhancedVoiceInterface(config.voice_enabled)
        self.pattern_analyzer = AdvancedPatternAnalyzer(self.memory)
        self.is_listening = False
        self.conversation_context = []
        self.user_name = "User"
        self._initialize_system()
    
    def _is_system_command(self, text: str) -> bool:
        """Check if input is a system-related command."""
        system_keywords = [
            'screenshot', 'screen capture', 'create file', 'make file',
            'open', 'launch', 'start', 'system info', 'system status',
            'minimize', 'maximize', 'close', 'focus', 'list windows', 'show windows',
            'app', 'program', 'task manager', 'settings', 'computer', 'pc'
        ]
        return any(keyword in text.lower() for keyword in system_keywords)
    
    def _initialize_system(self):
        """Initialize DexAI with enhanced personality."""
        logger.info("🤖 DexAI Advanced v2.0 - Initializing...")
        
        # Setup chrome driver for web automation
        if self.system_controller.chrome_driver:
            logger.info("✅ Web automation capabilities online.")
        else:
            logger.warning("⚠️ Web automation capabilities limited (ChromeDriver setup failed).")
        
        logger.info("🧠 Memory systems active.")
        logger.info("🎯 Pattern analysis ready.")
        logger.info("🎵 Media control enabled.")
        logger.info("📁 File management ready.")
        logger.info("🌐 Web automation available.")
        logger.info("\n🚀 DexAI Advanced is now online and ready to assist!")
        
        if self.voice_interface.enabled:
            welcome_message = f"Hello there! DexAI Advanced is now online. I'm your enhanced personal assistant, ready to help you with anything from controlling your computer to creating documents, playing music, and much more. Just say '{self.config.wake_word}' and I'll be right with you!"
            self.voice_interface.speak_with_personality(welcome_message, "friendly")
    
    async def process_advanced_input(self, user_input: str) -> str:
        """Process input with advanced AI and system integration."""
        user_input = user_input.strip()
        if not user_input:
            return "I didn't receive any input. Could you please say or type something?"
        
        try:
            # Analyze patterns first to get context and personality adjustments
            pattern_analysis = self.pattern_analyzer.analyze_advanced_patterns(user_input, "")
            response = ""
            
            # Handle different types of commands based on classification
            if self._is_system_command(user_input):
                response = await self._handle_advanced_system_command(user_input)
            elif self._is_web_command(user_input):
                response = await self._handle_web_command(user_input)
            elif self._is_media_command(user_input):
                response = await self._handle_media_command(user_input)
            elif self._is_office_command(user_input):
                response = await self._handle_office_command(user_input)
            else:
                # Generate AI response with enhanced context for general queries
                if self.gemini_api:
                    similar_conversations = self.memory.find_similar_conversations(user_input)
                    context = self._build_enhanced_context(similar_conversations, pattern_analysis)
                    response = await self.gemini_api.generate_enhanced_response(user_input, context)
                else:
                    response = "I'm sorry, but I don't have access to the AI service right now. I can still help you with system commands and basic tasks."
            
            # Store conversation with enhanced metadata
            self.memory.store_conversation(
                user_input,
                response,
                context=json.dumps(pattern_analysis),
                patterns=json.dumps(pattern_analysis.get('suggestions', []))
            )
            
            # Update conversation context for short-term memory
            self._update_conversation_context(user_input, response, pattern_analysis)
            return response
            
        except Exception as e:
            error_response = f"I encountered an unexpected issue: {str(e)}. Don't worry though, I'm still here to help! Try asking me something else or let me know if you'd like me to try a different approach."
            logger.error(f"Error in process_advanced_input: {e}")
            return error_response
    
    def _is_web_command(self, text: str) -> bool:
        """Check if input is a web-related command."""
        web_keywords = ['youtube', 'search youtube', 'play video', 'canva', 'open canva', 'web', 'browser', 'google', 'website', 'internet', 'visit', 'go to']
        return any(keyword in text.lower() for keyword in web_keywords)
    
    def _is_media_command(self, text: str) -> bool:
        """Check if input is a media command."""
        media_keywords = ['play music', 'spotify', 'pause', 'next track', 'previous', 'volume', 'music', 'song', 'audio', 'stop media', 'skip']
        return any(keyword in text.lower() for keyword in media_keywords)
    
    def _is_office_command(self, text: str) -> bool:
        """Check if input is an Office application command."""
        office_keywords = ['word', 'excel', 'powerpoint', 'document', 'spreadsheet', 'presentation', 'outlook', 'onenote', 'office']
        return any(keyword in text.lower() for keyword in office_keywords)
    
    async def _handle_web_command(self, command: str) -> str:
        """Handle web-related commands."""
        command_lower = command.lower()
        
        try:
            if 'youtube' in command_lower or 'play video' in command_lower:
                query_patterns = [
                    r'play\s+(?:video\s+)?(?:on\s+youtube\s+)?(.+?)(?:\s+on\s+youtube)?',
                    r'search\s+youtube\s+for\s+(.+)',
                    r'youtube\s+(.+)',
                    r'open\s+youtube\s+(.+)',
                    r'find\s+(.+)\s+on\s+youtube'
                ]
                query = None
                
                for pattern in query_patterns:
                    match = re.search(pattern, command_lower)
                    if match:
                        query = match.group(1).strip()
                        break
                
                if not query:
                    # Fallback: try to extract anything after common trigger words
                    trigger_words = ['play', 'search', 'youtube', 'video', 'find']
                    words = command_lower.split()
                    for i, word in enumerate(words):
                        if word in trigger_words and i < len(words) - 1:
                            query = ' '.join(words[i+1:])
                            break
                
                if query:
                    return self.system_controller.open_youtube_video(query)
                else:
                    return "🎬 What would you like me to search for or play on YouTube?"
            
            elif 'canva' in command_lower:
                project_types = ['document', 'presentation', 'poster', 'logo', 'instagram', 'flyer', 'video', 'social media']
                project_type = 'document'
                
                for ptype in project_types:
                    if ptype in command_lower:
                        project_type = ptype
                        break
                
                return self.system_controller.open_canva_project(project_type)
            
            elif any(browser in command_lower for browser in ['browser', 'chrome', 'firefox', 'edge']):
                url_match = re.search(r'(?:open|go to|visit)\s+(?:website\s+)?(https?://\S+|www\.\S+|\S+\.(?:com|org|net|gov|edu|io|dev|app))', command_lower)
                
                if url_match:
                    url = url_match.group(1)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    webbrowser.open(url)
                    return f"🌐 Opening {url} in your default browser."
                else:
                    webbrowser.open('https://www.google.com')
                    return "🌐 Opened your default browser to Google. What would you like to search for?"
            
            elif 'search' in command_lower:
                search_query_match = re.search(r'search\s+(?:for\s+)?(.+)', command_lower)
                if search_query_match:
                    query = search_query_match.group(1).strip()
                    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
                    return f"🌐 Searching Google for '{query}'."
                else:
                    return "🌐 What would you like me to search for?"
            
            else:
                return "🌐 I can help you with YouTube videos, Canva projects, opening specific websites, or general web searches. What would you like to do?"
                
        except Exception as e:
            logger.error(f"Web command failed: {e}")
            return f"❌ Web command failed: {str(e)}. Please ensure your browser is installed."
    
    async def _handle_media_command(self, command: str) -> str:
        """Handle media and music commands."""
        command_lower = command.lower()
        
        try:
            if 'spotify' in command_lower or 'play music' in command_lower:
                query_patterns = [
                    r'play\s+(?:music\s+)?(?:on\s+spotify\s+)?(.+?)(?:\s+on\s+spotify)?',
                    r'spotify\s+(.+)',
                    r'music\s+(.+)',
                    r'find\s+(.+)\s+on\s+spotify'
                ]
                query = None
                
                for pattern in query_patterns:
                    match = re.search(pattern, command_lower)
                    if match:
                        query = match.group(1).strip()
                        break
                
                if query:
                    return self.system_controller.play_spotify_music(query)
                else:
                    return "🎵 What music would you like me to play on Spotify?"
            
            elif any(action in command_lower for action in ['pause', 'play', 'next', 'previous', 'volume', 'mute', 'stop', 'skip']):
                # Media control actions
                if 'pause' in command_lower:
                    return self.system_controller.control_media('pause')
                elif 'play' in command_lower:
                    return self.system_controller.control_media('play')
                elif 'next' in command_lower or 'skip' in command_lower:
                    return self.system_controller.control_media('next')
                elif 'previous' in command_lower or 'prev' in command_lower:
                    return self.system_controller.control_media('previous')
                elif 'volume up' in command_lower:
                    return self.system_controller.control_media('volume_up')
                elif 'volume down' in command_lower:
                    return self.system_controller.control_media('volume_down')
                elif 'mute' in command_lower or 'unmute' in command_lower:
                    return self.system_controller.control_media('mute')
                elif 'stop' in command_lower:
                    return self.system_controller.control_media('stop')
                else:
                    return "🎵 I can control your media with play, pause, next, previous, volume up/down, or mute. What action would you like?"
            
            else:
                return "🎵 I can control your music - play, pause, next track, volume control, or search Spotify. What would you like?"
                
        except Exception as e:
            logger.error(f"Media command failed: {e}")
            return f"❌ Media command failed: {str(e)}. Please ensure a media player is active."
    
    async def _handle_office_command(self, command: str) -> str:
        """Handle Microsoft Office commands."""
        command_lower = command.lower()
        
        try:
            office_apps = ['word', 'excel', 'powerpoint', 'outlook', 'onenote']
            
            for app in office_apps:
                if app in command_lower:
                    create_new = 'new' in command_lower or 'create' in command_lower or 'blank' in command_lower
                    return self.system_controller.open_office_app(app, create_new)
            
            # Handle generic document/spreadsheet/presentation creation
            if 'document' in command_lower:
                return self.system_controller.open_office_app('word', True)
            elif 'spreadsheet' in command_lower:
                return self.system_controller.open_office_app('excel', True)
            elif 'presentation' in command_lower:
                return self.system_controller.open_office_app('powerpoint', True)
            
            return "📝 I can help you open Word, Excel, PowerPoint, Outlook, or OneNote. Which would you like to open, or would you like to create a new document/spreadsheet/presentation?"
            
        except Exception as e:
            logger.error(f"Office command failed: {e}")
            return f"❌ Office command failed: {str(e)}. Please ensure the Office application is installed."
    
    async def _handle_advanced_system_command(self, command: str) -> str:
        """Handle advanced system commands."""
        command_lower = command.lower()
        
        try:
            # File creation commands
            if 'create file' in command_lower or 'make file' in command_lower:
                filename_match = re.search(r'(?:create|make)\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+|named\s+)?(.+?)(?:\s+at\s+|$)', command_lower)
                location_match = re.search(r'at\s+([a-zA-Z]:[\\/](?:[^\\/]+[\\/])*[^\\/]*|[\\/](?:[^\\/]+[\\/])*[^\\/]*|desktop|documents|downloads)', command_lower)
                file_type_match = re.search(r'(?:a\s+)?(txt|py|html|js|css|pdf|doc|docx|xls|xlsx|ppt|pptx|json|xml|md)\s+file', command_lower)
                
                filename = filename_match.group(1).strip() if filename_match else "new_file"
                location = location_match.group(1).strip() if location_match else None
                file_type = file_type_match.group(1).strip() if file_type_match else "txt"
                
                # Resolve common locations
                if location:
                    if location.lower() == 'desktop':
                        location = os.path.join(os.path.expanduser("~"), "Desktop")
                    elif location.lower() == 'documents':
                        location = os.path.join(os.path.expanduser("~"), "Documents")
                    elif location.lower() == 'downloads':
                        location = os.path.join(os.path.expanduser("~"), "Downloads")
                
                # Clean filename from potential file type extensions
                known_extensions = ['.txt', '.py', '.html', '.js', '.css', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.json', '.xml', '.md']
                for ext in known_extensions:
                    if filename.lower().endswith(ext):
                        filename = filename[:-len(ext)]
                        break
                
                return self.system_controller.create_file_at_location(filename, location, file_type)
            
            # Screenshot commands
            elif 'screenshot' in command_lower or 'screen capture' in command_lower:
                location_match = re.search(r'at\s+([a-zA-Z]:[\\/](?:[^\\/]+[\\/])*[^\\/]*|[\\/](?:[^\\/]+[\\/])*[^\\/]*|desktop|documents|downloads)', command_lower)
                location = location_match.group(1).strip() if location_match else None
                
                # Resolve common locations for screenshot
                if location:
                    if location.lower() == 'desktop':
                        location = os.path.join(os.path.expanduser("~"), "Desktop")
                    elif location.lower() == 'documents':
                        location = os.path.join(os.path.expanduser("~"), "Documents")
                    elif location.lower() == 'downloads':
                        location = os.path.join(os.path.expanduser("~"), "Downloads")
                
                return self.system_controller.take_screenshot(location)
            
            # Window management
            elif any(action in command_lower for action in ['minimize', 'maximize', 'close', 'focus']):
                action = None
                app_name = None
                
                for act in ['minimize', 'maximize', 'close', 'focus']:
                    if act in command_lower:
                        action = act
                        break
                
                # Extract application name
                app_match = re.search(f'{action}\\s+(?:the\\s+)?(.+?)(?:\\s+window|\\s+app|\\s+program)?$', command_lower)
                if not app_match:
                    app_match = re.search(r'(.+?)(?:\\s+window|\\s+app|\\s+program)?\\s+(?:to\\s+)?' + action, command_lower)
                
                if app_match:
                    app_name = app_match.group(1).strip()
                else:
                    if "current window" in command_lower or "this window" in command_lower:
                        active_window = gw.getActiveWindow()
                        if active_window:
                            app_name = active_window.title
                        else:
                            return "❌ No active window to manage."
                
                if app_name:
                    return self.system_controller.manage_windows(action, app_name)
                else:
                    return f"❌ Please specify which window or application you'd like me to {action}."
            
            # List windows
            elif 'list windows' in command_lower or 'show windows' in command_lower or 'what windows are open' in command_lower:
                return self.system_controller.manage_windows('list')
            
            # System Info
            elif 'system info' in command_lower or 'system status' in command_lower or 'how is my computer doing' in command_lower:
                info = self.system_controller.get_system_info()
                if info:
                    return f"""🖥️ **System Status Report**
**Performance:**
• CPU Usage: {info['cpu_percent']:.1f}%
• Memory Usage: {info['memory']['percent']:.1f}%
• Available Memory: {info['memory']['available'] / (1024**3):.1f} GB
**Storage:**
• Disk Usage: {info['disk']['percent']:.1f}%
• Free Space: {info['disk']['free'] / (1024**3):.1f} GB
**System:**
• Active Processes: {info['running_processes']}
• Uptime: {info['uptime']/3600:.1f} hours
Everything looks good! Your system is running smoothly."""
                else:
                    return "❌ I couldn't retrieve system information. There might be an issue with accessing system metrics."
            
            # Open applications
            elif 'open' in command_lower or 'launch' in command_lower or 'start' in command_lower:
                app_match = re.search(r'(?:open|launch|start)\s+(?:the\s+)?(.+?)(?:\s+app|\s+program)?$', command_lower)
                if app_match:
                    app_name = app_match.group(1).strip()
                    if self.system_controller.open_application(app_name):
                        return f"✅ Opened {app_name} successfully!"
                    else:
                        return f"❌ I couldn't open {app_name}. Please check if it's installed or try a different name (e.g., 'VS Code' instead of 'code')."
                else:
                    return "❌ What application would you like me to open?"
            
            return "🤖 I can help with file creation, screenshots, window management, system info, and opening applications. What would you like me to do?"
            
        except Exception as e:
            logger.error(f"System command failed: {e}")
            return f"❌ System command failed: {str(e)}. Please ensure you have the necessary permissions."
    
    def _build_enhanced_context(self, similar_conversations: List[Dict], pattern_analysis: Dict) -> Dict[str, Any]:
        """Build enhanced context for AI response generation."""
        context = {
            'mood': pattern_analysis.get('mood', 'neutral'),
            'context': pattern_analysis.get('context', 'general'),
            'personality_adjustment': pattern_analysis.get('personality_adjustment', 'friendly'),
            'user_preferences': pattern_analysis.get('user_preferences', {}),
            'suggestions': pattern_analysis.get('suggestions', []),
            'recent_conversations': []
        }
        
        if similar_conversations:
            context['similar_conversations'] = similar_conversations[:2]
        
        if self.conversation_context:
            context['recent_conversations'] = self.conversation_context[-5:]
        
        return context
    
    def _update_conversation_context(self, user_input: str, response: str, pattern_analysis: Dict):
        """Update conversation context with enhanced metadata."""
        self.conversation_context.append({
            'user': user_input,
            'assistant': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'mood': pattern_analysis.get('mood', 'neutral'),
            'context': pattern_analysis.get('context', 'general'),
            'command_type': pattern_analysis.get('command_type', 'conversation')
        })
        
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)
    
    async def enhanced_voice_loop(self):
        """Enhanced voice interaction loop with better personality and error handling."""
        if not self.voice_interface.enabled:
            logger.warning("Voice interface is disabled or failed to initialize.")
            return
        
        self.is_listening = True
        welcome_msg = f"I'm listening for '{self.config.wake_word}' whenever you need me. Just speak naturally, and I'll help you with anything!"
        self.voice_interface.speak_with_personality(welcome_msg, "friendly")
        
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_listening and consecutive_failures < max_failures:
            try:
                logger.info(f"Waiting for wake word '{self.config.wake_word}'...")
                audio_input = self.voice_interface.listen(timeout=3, phrase_time_limit=3)
                
                if audio_input and self.config.wake_word in audio_input:
                    consecutive_failures = 0
                    
                    responses = [
                        "Yes, I'm here! What can I do for you?",
                        "At your service! How can I help?",
                        "Ready to assist! What do you need?",
                        "I'm listening! What would you like me to do?"
                    ]
                    import random
                    self.voice_interface.speak_with_personality(random.choice(responses), "friendly")
                    
                    logger.info("Listening for your command...")
                    command = self.voice_interface.listen(timeout=10, phrase_time_limit=8)
                    
                    if command:
                        logger.info(f"Processing: {command}")
                        
                        if len(command.split()) > 5:
                            self.voice_interface.speak_with_personality("Let me work on that for you...", "calm")
                        
                        response = await self.process_advanced_input(command)
                        logger.info(f"Response: {response}")
                        
                        emotion = "neutral"
                        if any(word in response.lower() for word in ['✅', 'success', 'opened', 'created', 'perfect', 'great']):
                            emotion = "friendly"
                        elif any(word in response.lower() for word in ['❌', 'error', 'failed', 'couldn\'t', 'issue']):
                            emotion = "calm"
                        elif any(word in response.lower() for word in ['🎵', '🎬', '🎨', 'excited', 'fantastic']):
                            emotion = "enthusiastic"
                        
                        self.voice_interface.speak_with_personality(response, emotion)
                    else:
                        self.voice_interface.speak_with_personality("I didn't catch your command. Could you please repeat that?", "calm")
                
                await asyncio.sleep(0.5)
                
            except sr.UnknownValueError:
                logger.warning("Voice recognition could not understand audio.")
                consecutive_failures += 1
                self.voice_interface.speak_with_personality("I'm sorry, I didn't quite understand that. Could you speak more clearly?", "calm")
                
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                consecutive_failures += 1
                self.voice_interface.speak_with_personality("I'm having trouble connecting to my speech service. Please check your internet connection.", "calm")
                
            except KeyboardInterrupt:
                self.is_listening = False
                self.voice_interface.speak_with_personality("DexAI signing off. It's been a pleasure assisting you today!", "friendly")
                break
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Voice loop error: {e}")
                
                if consecutive_failures < max_failures:
                    self.voice_interface.speak_with_personality("I had a small hiccup, but I'm still here and ready to help!", "calm")
                else:
                    self.voice_interface.speak_with_personality("I'm experiencing too many technical difficulties with the voice interface and will switch to text mode. Please try the text interface instead.", "calm")
                    self.is_listening = False
                
                await asyncio.sleep(2)
        
        if consecutive_failures >= max_failures:
            logger.warning("Voice interface encountered too many errors and will shut down.")
            await self.enhanced_text_interface()
    
    async def enhanced_text_interface(self):
        """Enhanced text-based interaction interface."""
        print("\n🤖 DexAI Advanced v2.0 - Text Interface")
        print("=" * 60)
        print("💡 I can help you with:")
        print("   🎵 Music control (Spotify, media keys)")
        print("   🎬 YouTube videos and web browsing")
        print("   📝 Office applications (Word, Excel, PowerPoint)")
        print("   🎨 Creative tools (Canva)")
        print("   📁 File and window management")
        print("   🖥️  System control and information")
        print("   💬 Natural conversation and assistance")
        print("-" * 60)
        print("Commands: 'quit', 'exit', 'bye' to stop | 'voice' for voice mode | 'help' for tips")
        print("=" * 60)
        
        while True:
            try:
                user_input = input(f"\n{self.user_name}: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    farewell_messages = [
                        "It's been great working with you today! Take care!",
                        "Thanks for letting me assist you! Have a wonderful day!",
                        "Until next time! I'll be here whenever you need me.",
                        "Goodbye! Remember, I'm always here to help make your digital life easier!"
                    ]
                    import random
                    print(f"🤖 DexAI: {random.choice(farewell_messages)}")
                    break
                
                if user_input.lower() == 'voice':
                    if self.voice_interface.enabled:
                        print("🎤 Switching to voice mode...")
                        await self.enhanced_voice_loop()
                        print("\nReturning to text interface.")
                        print("=" * 60)
                        print("💡 I can help you with:")
                        print("   🎵 Music control (Spotify, media keys)")
                        print("   🎬 YouTube videos and web browsing")
                        print("   📝 Office applications (Word, Excel, PowerPoint)")
                        print("   🎨 Creative tools (Canva)")
                        print("   📁 File and window management")
                        print("   🖥️  System control and information")
                        print("   💬 Natural conversation and assistance")
                        print("-" * 60)
                        print("Commands: 'quit', 'exit', 'bye' to stop | 'voice' for voice mode | 'help' for tips")
                        print("=" * 60)
                    else:
                        print("❌ Voice interface is not enabled or initialized in the configuration.")
                    continue
                
                if user_input.lower() == 'help':
                    help_text = """
🤖 **DexAI Advanced Help**
**Media & Entertainment:**
• "Play [song/artist] on Spotify" - Play music
• "Play [video] on YouTube" - Search and play videos
• "Pause", "Next track", "Volume up", "Stop media" - Media controls
**Office & Productivity:**
• "Open Word/Excel/PowerPoint/Outlook/OneNote" - Launch Office apps
• "Create new document/spreadsheet/presentation" - New Office file
• "Open Canva [document/presentation/poster]" - Design tools
**File Management:**
• "Create [txt/py/html] file [name] at [location]" - Create files (e.g., "create python file my_script at C:\\Users\\Desktop")
• "Take screenshot [at C:\\Screenshots]" - Capture screen
• "System info" - Check system status
**Window Control:**
• "Minimize/Maximize/Close/Focus [app name]" - Window management (e.g., "minimize Chrome")
• "List windows" - Show active windows
**Web Browsing:**
• "Open browser" / "Go to google.com" - Open web browser
• "Search [query]" - Search Google
**Natural Language:**
Just talk naturally! I understand context and can help with complex requests.
"""
                    print(help_text)
                    continue
                
                if user_input:
                    print("🤖 DexAI: Processing your request...")
                    response = await self.process_advanced_input(user_input)
                    print(f"🤖 DexAI: {response}")
                    
                    # Add helpful suggestions occasionally
                    if len(response) < 150 and not any(char in response for char in ['❌', 'Error', 'failed', 'couldn\'t', 'issue']):
                        import random
                        if random.random() > 0.7:
                            suggestions = self.pattern_analyzer._generate_smart_suggestions(user_input)
                            if suggestions:
                                print(f"\n💡 DexAI Tip: {random.choice(suggestions)}")
                            else:
                                generic_tips = [
                                    "\n💡 Tip: I can also help with voice commands - just type 'voice' to switch modes!",
                                    "\n💡 Tip: Try asking me to 'take a screenshot' or 'show system info' for quick tasks!",
                                    "\n💡 Tip: I can control your music, open applications, and manage files too!",
                                    "\n💡 Tip: You can ask me to 'list windows' to see what's open."
                                ]
                                print(f"🤖 DexAI: {random.choice(generic_tips)}")
                                
            except KeyboardInterrupt:
                print("\n🤖 DexAI: Goodbye! Thanks for using DexAI Advanced!")
                break
            except Exception as e:
                print(f"🤖 DexAI: I encountered an issue: {e}. Let's try something else!")
    
    async def run_advanced(self, interface_mode: str = "text"):
        """Run DexAI Advanced in specified mode."""
        try:
            if interface_mode == "voice":
                await self.enhanced_voice_loop()
            else:
                await self.enhanced_text_interface()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            print("DexAI Advanced encountered a critical error and will shut down.")

class VectorMemory:
    """Advanced vector database for conversation memory and pattern analysis."""
    
    def __init__(self, db_path: str, config: DexAIConfig):
        self.db_path = db_path
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.conversation_vectors = None
        self.conversations = []
        self._init_database()
        self._load_conversations()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    context TEXT,
                    patterns TEXT,
                    sentiment REAL,
                    topic TEXT,
                    command_type TEXT,
                    success_rating INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()
    
    def _load_conversations(self):
        """Load existing conversations from database and re-vectorize."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT user_input, ai_response, context, patterns, topic, command_type
                FROM conversations
                ORDER BY timestamp ASC
                LIMIT {self.config.max_memory_conversations}
            """)
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                self.conversations = rows
                
                # Prepare texts for vectorization
                texts = [f"{row[0]} {row[1]}" for row in self.conversations]
                
                if len(texts) > 1:
                    try:
                        self.conversation_vectors = self.vectorizer.fit_transform(texts)
                        logger.info(f"Loaded {len(self.conversations)} conversations and vectorized them.")
                    except Exception as e:
                        logger.error(f"Vector fitting error during load: {e}")
                        self.conversation_vectors = None
                else:
                    self.conversation_vectors = None
                    logger.warning("Not enough conversations to create meaningful vectors for similarity search.")
            else:
                self.conversations = []
                self.conversation_vectors = None
                logger.info("No conversations found in the database.")
                
        except sqlite3.Error as e:
            logger.error(f"Error loading conversations from database: {e}")
            self.conversations = []
            self.conversation_vectors = None
        finally:
            if conn:
                conn.close()
    
    def store_conversation(self, user_input: str, ai_response: str, context: str = "", patterns: str = ""):
        """Store conversation with enhanced metadata."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract topic and command type from context
        topic = 'general'
        command_type = 'conversation'
        try:
            context_dict = json.loads(context) if context else {}
            topic = context_dict.get('context', 'general')
            command_type = context_dict.get('command_type', 'conversation')
        except json.JSONDecodeError:
            logger.warning("Could not decode context JSON for storing conversation.")
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_input, ai_response, context, patterns, sentiment, topic, command_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, user_input, ai_response, context, patterns, 0.0, topic, command_type))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error storing conversation: {e}")
        finally:
            if conn:
                conn.close()
        
        # Re-load conversations to keep memory up-to-date
        self._load_conversations()
    
    def find_similar_conversations(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar past conversations using enhanced vector similarity."""
        if not self.conversations or self.conversation_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.conversation_vectors)[0]
            similar_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_conversations = []
            for idx in similar_indices:
                if similarities[idx] > self.config.similarity_threshold:
                    conv_data = self.conversations[idx]
                    similar_conversations.append({
                        'user_input': conv_data[0],
                        'ai_response': conv_data[1],
                        'context': conv_data[2],
                        'similarity': float(similarities[idx]),
                        'topic': conv_data[4],
                        'command_type': conv_data[5]
                    })
            return similar_conversations
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

async def main_advanced():
    """Main function to run DexAI Advanced."""
    print("🚀 DexAI Advanced v2.0 - Initialization")
    print("=" * 50)
    
    # Configuration
    config = DexAIConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        voice_enabled=True,
        wake_word="hey dex",
        max_memory_conversations=5000,
        chrome_driver_path="",
        default_file_location=str(Path.home() / "Desktop")
    )
    
    if not config.gemini_api_key:
        print("\n⚠️  WARNING: Gemini API Key is not set!")
        print("Please set the 'GEMINI_API_KEY' environment variable or add it to a .env file.")
        print("DexAI will operate in a limited capacity without the API key for AI responses.")
    
    # Initialize DexAI Advanced
    print("\n🤖 Initializing DexAI Advanced...")
    dex = DexAIAdvanced(config)
    
    # Choose interface mode
    print("\n🎯 Choose your interaction mode:")
    print("1. 💬 Text Interface (recommended for first use)")
    print("2. 🎤 Voice Interface (requires microphone and PyAudio)")
    print("3. 🔄 Auto (start with text, switch to voice anytime)")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            await dex.run_advanced("text")
            break
        elif choice == "2":
            if config.voice_enabled and dex.voice_interface.enabled:
                await dex.run_advanced("voice")
                break
            else:
                print("❌ Voice interface is disabled or failed to initialize. Please ensure PyAudio is installed and your microphone is working, or choose option 1 or 3.")
        elif choice == "3":
            await dex.run_advanced("text")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    print("🌟 Welcome to DexAI Advanced v2.0")
    print("Your Enhanced Personal AI Assistant")
    print("=" * 50)
    
    try:
        asyncio.run(main_advanced())
    except KeyboardInterrupt:
        print("\n👋 DexAI Advanced shutdown complete. Goodbye!")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)