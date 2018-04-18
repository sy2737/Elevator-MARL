from env import make
import time

if __name__=="__main__":
    env = make(1, 10, [60]*10, 135, 1200, 5, 1)
    s = env.reset()
    while True:
        print("Current state:", s)
        print("Taking some action...")
        s = env.step(None)
        time.sleep(0.5)
