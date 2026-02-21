import pygame
import os

class AlertSystem:
    def __init__(self, sound_path):
        self.enabled = False
        # Set dummy drivers so pygame doesn't look for a screen or speakers
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        
        if os.path.exists(sound_path):
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(sound_path)
                self.enabled = True
            except Exception as e:
                print(f"⚠️ Audio initialization failed: {e}")
        else:
            print(f"⚠️ Alarm sound not found at {sound_path}")

    def play(self):
        if self.enabled:
            try:
                self.sound.play()
            except:
                pass
