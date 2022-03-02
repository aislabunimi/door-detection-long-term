from gibson.envs.no_physiscs_env import TurtlebotNavigateNoPhysicsEnv
import argparse

from doors_detection_long_term.positions_extractor.gibson_run_config import GibsonConfigRun

config_file = GibsonConfigRun(simulation_env=TurtlebotNavigateNoPhysicsEnv, world_name='house1', floor=0)\
    .is_discrete(False).remove_semantics().write_to_file()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotNavigateNoPhysicsEnv(config=args.config)
    env.reset()
    env.robot.set_position([1, 0, 1])
    while True:
        # Remember to call env.step before every robot position change
        env.step([0.0, 0.0])
        env.robot.turn_left(delta=0.1)

        # Now you can move the robot in space without constraints (remember to call again env.step)
        #env.robot.set_position([1, 0, 1])
        #env.robot.move_forward(forward=1)
        #env.robot.move_backward(backward=0.2)

        #env.robot.turn_right(delta=0.1)
