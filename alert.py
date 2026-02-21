import pygame
import os

class AlertSystem:
    def __init__(self, sound_path):
        self.enabled = False
        
        # Check if we are running in a headless environment (like Streamlit Cloud)
        # This prevents the "No such audio device" crash
        if os.environ.get('SDL_VIDEODRIVER') == 'dummy' or os.path.exists('/.dockerenv'):
            os.environ['SDL_AUDIODRIVER'] = 'dummy'

        if os.path.exists(sound_path):
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(sound_path)
                self.enabled = True
            except Exception as e:
                print(f"⚠️ Could not initialize audio: {e}")
        else:
            print("⚠️ Alarm sound not found, alerts will be silent")

    def play(self):
        if self.enabled:
            self.sound.play()
