import copy
import os
from typing import Type, Dict

import yaml
from gibson.envs.env_modalities import BaseRobotEnv
from termcolor import colored

from doors_detection_long_term.positions_extractor.gibson_environments_data import GibsonEnvironmentsData


class EnvironmentNotSemanticallyAnnotatedException(Exception):
    pass


class GibsonConfigRun:
    def __init__(self, simulation_env: Type[BaseRobotEnv], world_name: str, floor: int):
        """
        Creates new instance of GibsonConfigRun.
        This methods automatically sets the starting position and orientation according to the values specified in the environments_data.yaml file.
        If an environment is semantically annotated, these information are rendered and shown to the user.
        To remove them, use the method 'remove_semantics'
        :param world_name: the name of the environment
        :type world_name: str
        :param floor: the floor where to place the robot
        :type floor: int
        :param simulation_env: the simulation environment to use in Gibson
        """

        self._gibson_environments_data = GibsonEnvironmentsData()
        with open(os.path.join(os.path.dirname(__file__), 'data', 'gibson_config_file.yaml'), mode='r') as config_file:
            self._gibson_config_parameters = yaml.load(config_file, Loader=yaml.FullLoader)

        # Set the simulation environment
        self._gibson_config_parameters['envname'] = simulation_env.__name__

        self._env_name = world_name

        env_data = self._gibson_environments_data.get_environment_data(env_name=world_name)
        self._gibson_config_parameters['model_id'] = world_name
        self._gibson_config_parameters['initial_pos'] = env_data[GibsonEnvironmentsData.KEY_FLOORS][floor][GibsonEnvironmentsData.KEY_POSITION]
        self._gibson_config_parameters['initial_orn'] = env_data[GibsonEnvironmentsData.KEY_FLOORS][floor][GibsonEnvironmentsData.KEY_ORIENTATION]

        # Remove semantics if the environment is not semantically annotated
        if not env_data[GibsonEnvironmentsData.KEY_HAS_SEMANTICS]:
            self.remove_semantics()
        else:
            if env_data[GibsonEnvironmentsData.KEY_DATASET] == 'stanford':
                self._gibson_config_parameters['semantic_source'] = 1
                self._gibson_config_parameters['semantic_color'] = 3
            elif env_data[GibsonEnvironmentsData.KEY_DATASET] == 'matterport':
                self._gibson_config_parameters['semantic_source'] = 2
                self._gibson_config_parameters['semantic_color'] = 2

    def is_discrete(self, discrete: bool) -> 'GibsonConfigRun':
        """
        Sets the discrete parameters. It must be True if the simulator is used with the PLAY utility, otherwise it must be False.
        :param discrete: the paramater value
        :return: GibsonConfigRun
        """
        self._gibson_config_parameters['is_discrete'] = discrete
        return self

    def remove_semantics(self) -> 'GibsonConfigRun':
        """
        If an environment is semantically annotated, the semantic data are automatically synthesized and shown.
        This method tells Gibson not to produce semantic data.
        If there are no semantic data for the given environment, this method has no effect.
        :return: GibsonConfigRun
        """
        self._gibson_config_parameters['output'] = ['nonviz_sensor', 'rgb_filled', 'depth']
        self._gibson_config_parameters['ui_components'] = ['RGB_FILLED', 'DEPTH']
        self._gibson_config_parameters['ui_num'] = 2
        return self

    def set_semantics_to_random_color(self) -> 'GibsonConfigRun':
        """
        This method sets the rendering of semantic information to 'Instance-by-Instance Color Coding' mode.
        This modality consists in assigning a random distinctive color to each semantic instance.
        These colors are arbitrarily chosen but are preserved through different trials.
        Note that this mode renders intuitive colorful semantic map frames, but the rgb values do not enable easy semantic class lookup.
        :raise EnvironmentNotSemanticallyAnnotatedException if the environment is not semantically annotated
        :return: GibsonConfigRun
        """
        env_data = self._gibson_environments_data.get_environment_data(env_name=self._env_name)

        if not env_data[GibsonEnvironmentsData.KEY_HAS_SEMANTICS]:
            print(colored('The environment \'{0}\' is not semantically annotated, you cannot show semantic data!!'.format(self._env_name), 'red'))
            raise EnvironmentNotSemanticallyAnnotatedException()

        self._gibson_config_parameters['semantic_color'] = 1

        return self

    def set_semantic_labels_to_color(self) -> 'GibsonConfigRun':
        """
        This method sets the rendering of semantic information to 'Semantic Label Color Coding' mode.
        Using this semantic modality, the environment assigns a semantic label to each object category and each label is converted
        to the corresponding RGB color using this formula:
        b = ( label ) % 256;
        g = ( label >> 8 ) % 256;
        r = ( label >> 16 ) % 256;
        color = {r, g, b}
        Rendered colors usually look dark to human eyes, but rendered frames can be consumed directly
        as semantically tagged pixel maps.
        :raise EnvironmentNotSemanticallyAnnotatedException if the environment is not semantically annotated
        :return: GibsonConfigRun
        """
        env_data = self._gibson_environments_data.get_environment_data(env_name=self._env_name)

        if not env_data[GibsonEnvironmentsData.KEY_HAS_SEMANTICS]:
            print(colored('The environment \'{0}\' is not semantically annotated, you cannot show semantic data!!'.format(self._env_name), 'red'))
            raise EnvironmentNotSemanticallyAnnotatedException()

        if env_data[GibsonEnvironmentsData.KEY_DATASET] == 'stanford':
            self._gibson_config_parameters['semantic_color'] = 3
        elif env_data[GibsonEnvironmentsData.KEY_DATASET] == 'matterport':
            self._gibson_config_parameters['semantic_color'] = 2

        return self

    def write_to_file(self) -> str:
        """
        Writes the specified configuration in a temporary file and returns its path.
        :return: the file's path which contains the set configuration
        """
        save_path = os.path.join(os.path.dirname(__file__), 'data', 'gibson_config_file_temp.yaml')
        with open(save_path, mode='w') as gibson_config_file_temp:
            yaml.dump(self._gibson_config_parameters, gibson_config_file_temp, default_flow_style=False)

        return save_path

    def get_parameters(self) -> Dict:
        """
        Returns the configuration parameters set with this class
        :return: a dictionary with the configuration parameters
        """
        return copy.deepcopy(self._gibson_config_parameters)