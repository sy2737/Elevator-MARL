from env import make
import time

if __name__=="__main__":
    env = make(1, 10, [1/60]*10, 135, 1200, 5, 1)
    s = env.reset()

    action = 4 # switches between 4, and 6, corresponding to idle_up_idle and idle_down_idle
    while True:
        print("Taking action {}".format(action))
        s = env.step(action)
        env.render()
        if s[-1][0] in [0, 9]:
            if action == 4:
                action = 6
            else:
                action = 4
        time.sleep(0.5)
