from Model import AttrDict
from utils.mypaint_agent import MypaintPainter


class Agent:
    def __init__(self, brush_config):
        self.painter = MypaintPainter(brush_config)

    def execute(self, waypoints):
        """
        Execute a complete trajectory
        Args:
            waypoints: (list) list of points (x,y,z) in float between 0-1

        Returns:

        """
        self.painter.reset()
        for point in waypoints:
            self.painter.paint(point[0], point[1], point[2])
        return self.painter.get_img()


def test_agent():
    import matplotlib.pyplot  as plt

    config = AttrDict()
    config.brush_name = 'custom/slow_ink'
    config.agent_name = 'Physics'
    robot = Agent(config.brush_config)
    img = robot.execute([(0.5, 0.5, 0),
                         (0.5, 0.5, 1),
                         (0.8, 0.8, 1),
                         (0.2, 0.8, 1),
                         (0.8, 0.2, 1),
                         ])
    plt.imshow(img)
    plt.show()
