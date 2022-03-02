from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
import argparse

from gibson.utils.play import play

from doors_detection_long_term.positions_extractor.gibson_run_config import GibsonConfigRun

# Matterport world without semantic
#config_file = GibsonConfigRun(simulation_env=TurtlebotNavigateEnv, world_name='house1', floor=0).remove_semantics().is_discrete(True).write_to_file()

# Matterport world with semantic information (Semantic Label to Color Coding)
#config_file = GibsonConfigRun(simulation_env=TurtlebotNavigateEnv, world_name='house1', floor=0).is_discrete(True).write_to_file()

# Stanford environment with semantic data (Instance-by-Instance Color Coding)
config_file = GibsonConfigRun(simulation_env=TurtlebotNavigateEnv, world_name='space7', floor=0).is_discrete(True).set_semantics_to_random_color().write_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotNavigateEnv(config=config_file)
    print(env.config)
    play(env, zoom=4)