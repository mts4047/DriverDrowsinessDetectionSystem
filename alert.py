import pygame
import os

class AlertSystem:
    def __init__(self, sound_path):
        self.enabled = False
        if os.path.exists(sound_path):
            try:
                # Tell pygame to use a dummy audio driver for headless servers
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                os.environ['SDL_AUDIODRIVER'] = 'dummy'
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(sound_path)
                self.enabled = True
            except Exception as e:
                print(f"Audio initialization failed: {e}")
        else:
            print("⚠️ Alarm sound not found")

    def play(self):
        if self.enabled:
            try:
                self.sound.play()
            except:
                pass
