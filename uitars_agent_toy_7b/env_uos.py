"""
get_obs  ### 返回截屏
step(action) 执行pyautogui



"""
import os
import time

os.environ["DISPLAY"] = ":0"

from pyautogui import screenshot, click, write, scroll, hotkey
from PIL import Image
import os
import datetime
import subprocess

def take_screenshot(save_path):
    import os
    os.environ["DISPLAY"] = ":0"
    subprocess.run(['scrot', save_path], check=True)


class Env:
    def __init__(self, screenshot_dir="/mnt/data/screenshots"):
        self.screenshot_dir = screenshot_dir
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    # def _get_obs(self):
    #     # Take a screenshot and save it
    #     img = screenshot()
    #     screenshot_path = os.path.join(self.screenshot_dir, datetime.datetime.now().strftime("%Y%m%d@%H%M%S")+'.png')
    #     img.save(screenshot_path)
    #     obs = {'screenshot': screenshot_path}
    #     time.sleep(3)
    #     return obs
    def _get_obs(self):
        screenshot_path = os.path.join(self.screenshot_dir, datetime.datetime.now().strftime("%Y%m%d@%H%M%S") + '.png')
        take_screenshot(screenshot_path)
        obs = {'screenshot': screenshot_path}
        time.sleep(1)
        return obs

    def step(self, action, **kwargs):### pyautogui
        print('ENV ....',action)### python code包括 pyautogui执行的
        obs, reward, done, info=None,None,None,None
        exec(action)#### 执行 pyautogui
        time.sleep(2)
        obs=self._get_obs()
        return obs, reward, done, info
        # Perform actions using pyautogui
        # if action == "click":
        #     x, y = kwargs.get("x"), kwargs.get("y")
        #     click(x, y)
        # elif action == "write":
        #     text = kwargs.get("text")
        #     write(text)
        # elif action == "scroll":
        #     amount = kwargs.get("amount")
        #     scroll(amount)
        # elif action == "hotkey":
        #     keys = kwargs.get("keys")
        #     hotkey(*keys)
        # else:
        #     raise ValueError("Unsupported action")
    def reset(self):
        print('reset env')


if __name__=='__main__':
    # Example usage:
    # env_instance = env()
    # print(env_instance.get_obs())
    # env_instance.step("click", x=100, y=200)
    # env_instance.step("write", text="Hello")
    # env_instance.step("scroll", amount=200)
    # env_instance.step("hotkey", keys=["ctrl", "c"])

    # Note: The example usage is commented out to prevent actual execution.

    env_instance = Env(screenshot_dir='./tmp1')  # Create an instance to test the directory creation
    #env_instance.screenshot_dir  # Return the directory path to check if it's correctly set up
    env_instance._get_obs()
