import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.config import Config

Config.set('graphics', 'width', '540')
Config.set('graphics', 'height', '960')  # 16:9

class flower_ui(App):
    def build(self):
        return super().build()
    

if __name__=='__main__':
    flower_ui().run()