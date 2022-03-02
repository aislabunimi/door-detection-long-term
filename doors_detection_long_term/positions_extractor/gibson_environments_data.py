import os.path
import copy
from typing import Dict, KeysView

import yaml


class GibsonEnvironmentsData:
    KEY_HAS_SEMANTICS = 'has_semantics'
    KEY_DATASET = 'dataset'
    KEY_FLOORS = 'floors'
    KEY_POSITION = 'position'
    KEY_ORIENTATION = 'orientation'
    KEY_FLOOR_HEIGHT = 'floor_height'

    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'data', 'environments_data.yaml'), mode='r') as f:
            self._environments_data: Dict = yaml.load(f, Loader=yaml.FullLoader)

    def get_environment_data(self, env_name: str) -> Dict:
        """
        The data relative to the given environment
        :param env_name: the name of the environment
        :type env_name: str
        :return: the data of the environment with the given name
        :rtype: Dict
        """
        return copy.deepcopy(self._environments_data[env_name])

    def get_environments_data(self) -> Dict:
        """
        Returns the environments' data
        :return: the environments' data
        :rtype: Dict
        """
        return copy.deepcopy(self._environments_data)

    def get_environments_with_semantics(self) -> Dict:
        """
        Returns only the environments' data that are semantically annotated
        :return:
        :rtype: Dict
        """
        return copy.deepcopy({key: value for key, value in self._environments_data.items() if value[GibsonEnvironmentsData.KEY_HAS_SEMANTICS]})

    def get_env_names(self) -> KeysView:
        """
        Returns the environments' names
        :return: the environments' names
        :rtype: KeysView[str]
        """
        return self._environments_data.keys()
