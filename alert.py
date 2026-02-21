import pygame
import os

class AlertSystem:
    def __init__(self, sound_path):
        self.enabled = False
        if os.path.exists(sound_path):
            try:
                # Attempt to initialize the mixer
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(sound_path)
                self.enabled = True
            except pygame.error as e:
                # This will happen on Streamlit Cloud
                print(f"⚠️ Audio device not found: {e}. Alerts will be visual only.")
        else:
            print("⚠️ Alarm sound file not found.")

    def play(self):
        if self.enabled:
            try:
                self.sound.play()
            except Exception:
                pass # Prevent crash if play fails
