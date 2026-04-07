from GPU.env.game2048_env import Game2048Env

env = Game2048Env()
obs, info = env.reset()
done = False

print("Initial Board:")
env.render()

for _ in range(10):
    action = env.action_space.sample() # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    env.render()
    if terminated:
        print("Game Over")
        break