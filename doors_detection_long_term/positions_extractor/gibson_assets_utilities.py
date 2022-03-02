import os
from functools import reduce
from typing import Tuple, Dict, Union

import cv2
import yaml
from matplotlib.figure import Figure
import numpy as np
import trimesh
from gibson.assets.assets_manager import AssetsManager
from matplotlib import pyplot as plt
from termcolor import colored

from doors_detection_long_term.positions_extractor.gibson_environments_data import GibsonEnvironmentsData


class GibsonAssetsUtilities:
    def __init__(self):
        self._assets_path = AssetsManager().get_assets_path()
        self._environments_data = GibsonEnvironmentsData()

    def load_obj(self, env_name: str) -> trimesh.Trimesh:
        """
        Loads and returns the specified environment obj file
        :param env_name: the name of the environment whose wavefront file is to be loaded
        :type env_name: str
        :return: the mesh stored in the obj file
        :rtype: trimesh.Trimesh
        """

        obj_path = os.path.join(self._assets_path, 'dataset', env_name, 'mesh_z_up.obj')
        if not os.path.exists(obj_path):
            print(colored('The specified object file does not exists!!', 'red'))
            raise FileNotFoundError(obj_path)

        mesh = trimesh.load_mesh(obj_path, file_type='obj')

        # If the loaded onj file generates a scene, the meshes inside it must be concatenated
        if isinstance(mesh, trimesh.Scene):
            mesh = reduce(lambda mesh1, mesh2: trimesh.util.concatenate(mesh1, mesh2), list(mesh.geometry.values()))

        return mesh

    @staticmethod
    def GET_FILE_NAME(env_name: str, floor: int) -> str:
        return env_name + '_floor_' + str(floor)

    def create_floor_map(self, env_name: str, floor: int, image_size: Union[Tuple[int, int], str] = 'auto', floor_offset: float = 0.10, height: float = 1.0, step: float = 0.10, save_to_file: bool = False):
        """
        Generates the map of the environment at the given floor and the relative metadata.
        The floor map is a png image, while metadata indicates:
            1) the pixel coordinates to the origin point (0.0, 0.0)
            2) the scale to map a pixel to the real-world distance it covers
        To generate the map, the mesh is sliced at multiple heights.
        The cuts begin from the floor offset and they are made at each step up to the maximum height.
        :param env_name:
        :param floor:
        :param floor_offset: the offset to start cutting the mesh (the first cross section is performed at [floor_height + floor_offset]
        :type floor_offset: float
        :param image_size: the size of the map image in pixel. The default value is 'auto':
                                it is a applied an easy algorithm that automatically set the image size based on environments' dimensions
        :type image_size: Tuple[int, int] or str
        :param height: the maximum height to stop cutting the mesh. This means that the last mesh cut has is made at [floor_height + floor_offset + height]
        :type height: float
        :param step: the step used to cut the environment's mesh
        :type step: float
        :param save_to_file: all generated maps and relatives metadata are saved in temporary map folder. To save them permanently, set this flag True
        :type save_to_file: bool
        :return: None
        """
        mesh = self.load_obj(env_name=env_name)

        floor_height = self._environments_data.get_environment_data(env_name)[GibsonEnvironmentsData.KEY_FLOORS][floor][GibsonEnvironmentsData.KEY_FLOOR_HEIGHT]
        floor_height += floor_offset
        slices_2D = mesh.section_multiplane(plane_origin=[0, 0, floor_height + floor_offset], plane_normal=[0, 0, 1], heights=np.arange(0.0, height, step).tolist())

        plt.close()
        plt.axis('off')

        for slice in slices_2D:
            slice.plot_entities(show=False, annotations=False, color='k')

        fig: Figure = plt.gcf()
        fig.tight_layout()

        if image_size == 'auto':
            ymin, ymax = plt.gca().get_ylim()
            xmin, xmax = plt.gca().get_xlim()
            longer = abs(ymin - ymax) if abs(ymin - ymax) > abs(xmin - xmax) else abs(xmin - xmax)
            fig.set_size_inches(longer * 30 / 100.0, longer * 30 / 100.0)
        else:
            fig.set_size_inches(image_size[0] / 100.0, image_size[1] / 100.0)

        fig.canvas.draw()

        # Extract metadata
        ax = fig.gca()
        _, height = fig.canvas.get_width_height()
        x_0, y_0 = ax.transData.transform((0.0, 0.0))

        inv_x_0, _ = ax.transData.transform((0, 0))
        inv_x_1, _ = ax.transData.transform((1, 0))
        scale = 1 / abs(inv_x_0 - inv_x_1)
        metadata = {'origin': {'x': int(round(x_0)), 'y': int(round(height - y_0))}, 'scale': float(round(scale, 6))}

        # Save files in temporary folder
        file_name = GibsonAssetsUtilities.GET_FILE_NAME(env_name, floor)
        fig.savefig(os.path.join(os.path.dirname(__file__), 'data', 'temporary_maps', 'maps', file_name + '.png'), dpi=fig.dpi)
        with open(os.path.join(os.path.dirname(__file__), 'data', 'temporary_maps', 'maps_metadata', file_name + '.yaml'), mode='w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        if save_to_file:
            file_name = GibsonAssetsUtilities.GET_FILE_NAME(env_name, floor)
            fig.savefig(os.path.join(os.path.dirname(__file__), 'data', 'maps', file_name + '.png'), dpi=fig.dpi)
            with open(os.path.join(os.path.dirname(__file__), 'data', 'maps_metadata', file_name + '.yaml'), mode='w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

    def load_map_and_metadata(self, env_name: str, floor: int) -> Tuple[np.array, Dict]:
        """
        Loads form disk the map and its metadata of the specified environment
        :param env_name: the environment name
        :param floor: the environment floor
        :return: the map and the relative metadata
        """
        map = cv2.imread(os.path.join(os.path.dirname(__file__), 'data', 'maps', GibsonAssetsUtilities.GET_FILE_NAME(env_name, floor) + '.png'))
        with open(os.path.join(os.path.dirname(__file__), 'data', 'maps_metadata', GibsonAssetsUtilities.GET_FILE_NAME(env_name, floor)) + '.yaml', mode='r') as f:
            metadata: Dict = yaml.load(f, Loader=yaml.FullLoader)

        return map, metadata
