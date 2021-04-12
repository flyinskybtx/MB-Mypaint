from Env.agent import Agent
from Main import load_config

if __name__ == '__main__':
    config = load_config()
    robot = Agent(config.brush_config)

