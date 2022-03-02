import sys
from typing import Union, Tuple, List, Dict, Set
import numpy as np


sys.setrecursionlimit(10000)


class Coordinate:
    """
    This class represents a coordinate in a image (pixel indexes) or in the real world.
    By convention, in an image x and y are:
        - x = row index
        - y = column index
    A pixel in an image is column first indexed (pixel=img[y, x]).
    A point in real world is expressed as follow: point=real[x, y].
    Note that the image and real world indexes are inverted.
    This class automates this simple but error-prone convention and
    it offers the utility methods to convert a pixel coordinate to a real world point and vice versa.
    """

    def __init__(self, x: Union[int, float], y: Union[int, float], map_origin: Tuple[int, int], scale: float):
        """
        Instantiates a new coordinate. x and y must have the same type.
        If x and y are integer, the coordinate is assumed to be a pixel index,
        otherwise, if x and y are float, the instance is a real represent a point in the real world.
        To perform conversion operations, additional information are necessary: the map origin (the coordinate of the real world origin in the image)
        and the scale (the distance in meter covered by a pixel).
        """
        self._x = x
        self._y = y

        if type(x) != type(y):
            raise TypeError('The x and y coordinates must have the same type, int or float!!!')

        self._map_origin = map_origin
        self._scale = scale

    def is_real_coordinate(self):
        return isinstance(self._x,  float) and isinstance(self._y, float)

    def is_image_coordinate(self):
        return isinstance(self._x, int) and isinstance(self._y, int)

    def _convert_to_image_coordinate_tuple(self):
        if self.is_image_coordinate():
            return self._y, self._x
        elif self.is_real_coordinate():
            x_img = int(round(self._x / self._scale)) + self._map_origin[0]
            y_img = self._map_origin[1] - int(round(self._y / self._scale))
            return y_img, x_img

    def _convert_to_real_coordinate_tuple(self):
        if self.is_real_coordinate():
            return self._x, self._y
        elif self.is_image_coordinate():
            x_real = (self._x - self._map_origin[0]) * self._scale
            y_real = (self._map_origin[1] - self._y) * self._scale
            return x_real, y_real

    def to_img_index(self) -> Tuple[int, int]:
        """
        Returns a tuple which can be used to index a pixel (y, x).
        :return: the image coordinate (pixel[y, x])
        """
        return self._convert_to_image_coordinate_tuple()

    def to_real_point(self) -> Tuple[float, float]:
        """
        Returns a tuple of float which can be used as coordinate in real world (x, y).
        :return:
        """
        return self._convert_to_real_coordinate_tuple()

    def get_x_y_tuple(self):
        """
        Returns the x and y values as tuple
        """
        return self._x, self._y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return isinstance(other, Coordinate) and \
               (other.is_image_coordinate() and self.is_image_coordinate() or other.is_real_coordinate() and self.is_real_coordinate()) and \
               other._x == self._x and other._y == self._y

    def __hash__(self):
        return hash((self._x, self._y))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        y_img, x_img = self._convert_to_image_coordinate_tuple()
        x_real, y_real = self._convert_to_real_coordinate_tuple()
        return 'Coordinate: img[{0}, {1}] -> real[{2}, {3}]'.format(x_img, y_img, x_real, y_real)


class Node:
    """
    This class represents a graph node.
    """
    def __init__(self, coordinate: Coordinate):
        self._coordinate = coordinate
        self._connected_nodes: List[Node] = []

    def connect_with(self, node: 'Node'):
        if node not in self._connected_nodes:
            self._connected_nodes.append(node)

    def get_coordinate(self):
        return self._coordinate

    def get_connected_nodes(self) -> List['Node']:
        return self._connected_nodes

    def remove_connected_node(self, node: 'Node'):
        self._connected_nodes.remove(node)

    def get_connected_nodes_count(self) -> int:
        return len(self._connected_nodes)

    def __eq__(self, other):
        return isinstance(other, Node) and self._coordinate.__eq__(other._coordinate)

    def __hash__(self):
        return self._coordinate.__hash__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        y_img, x_img = self._coordinate.to_img_index()
        x_real, y_real = self._coordinate.to_real_point()
        return 'Node: img[{0}, {1}] -> real[{2}, {3}]'.format(x_img, y_img, x_real, y_real)


class Graph:
    def __init__(self, image_width: int, image_height: int, map_origin: Coordinate, scale: float):
        self._image_width = image_width
        self._image_height = image_height
        self._nodes: Dict[Coordinate, Node] = {}
        self._arcs = set()
        self._connected_components: Dict[int, Set[Node]] = {}
        self._map_origin = map_origin
        self._scale = scale

    def add_node(self, node: Node):
        if node.get_coordinate() not in self._nodes:
            self._nodes[node.get_coordinate()] = node

    def get_nodes(self) -> Dict[Coordinate, Node]:
        return self._nodes

    def add_connection(self, node1_coordinates: Coordinate, node2_coordinates: Coordinate):
        if not node1_coordinates.is_image_coordinate() or not node2_coordinates.is_image_coordinate():
            raise TypeError('The point coordinates must be image coordinates!!')
        self._nodes[node1_coordinates].connect_with(self._nodes[node2_coordinates])

    def get_connected_components(self):
        return self._connected_components

    def get_graph_bitmap(self):
        graph_bitmap = np.array(
            [255 for _ in range(self._image_width * self._image_height)], dtype=np.uint8
        ).reshape((self._image_width, self._image_height))

        for coordinate in self._nodes.keys():
            graph_bitmap[coordinate.to_img_index()] = 0

        return graph_bitmap

    def prune_side_lines(self):
        visited_nodes = set()

        def iterate(node: Node, component: int):
            if node in visited_nodes:
                return

            visited_nodes.add(node)

            for connected_node in node.get_connected_nodes():
                iterate(connected_node, component)

            remove_nodes = set()
            for connected_node in node.get_connected_nodes():
                if len(connected_node.get_connected_nodes()) == 1:
                    remove_nodes.add(connected_node)

            for remove_node in remove_nodes:
                node.get_connected_nodes().remove(remove_node)
                self._nodes.pop(remove_node.get_coordinate())
                self._connected_components[component].remove(remove_node)

        for component in self._connected_components.keys():
            visited_nodes = set()
            node = list(self._connected_components[component])[0]
            iterate(node, component)

    def find_connected_components(self):
        component_id = 0

        visited_nodes = set()

        nodes_in_components: Dict[Node, int] = {}

        def find_connected_component(node: Node):
            if node in visited_nodes:
                return

            visited_nodes.add(node)
            self._connected_components[component_id].add(node)
            nodes_in_components[node] = component_id

            for connected_node in node.get_connected_nodes():
                find_connected_component(connected_node)

        for node in self._nodes.values():

            # Verify that the node does not already belong to a connected component
            if node not in nodes_in_components:
                visited_nodes = set()
                self._connected_components[component_id] = set()
                find_connected_component(node)
                component_id += 1

    def get_map_origin(self) -> Coordinate:
        return self._map_origin

    def get_scale(self) -> float:
        return self._scale

    def get_real_position(self, interval: float) -> List[Coordinate]:
        """
        Returns a list of real positions through the graph, considering each connected component.
        The positions are sampled with a distance specified by interval parameter.
        :param interval: the distance between two consecutive positions in meter
        :return: a list containing the real positions
        """
        visited_nodes: Set[Node] = set()
        positions: List[Coordinate] = []

        def find_positions(node: Node, max_distance: float, distance: float):
            if node in visited_nodes:
                return

            visited_nodes.add(node)
            for connected_node in node.get_connected_nodes():
                # Discard a previously visited node
                if connected_node in visited_nodes:
                    continue

                # Create the points that define the segment, create the vector and find its length
                current_distance = distance
                point_a = node.get_coordinate().to_real_point()
                point_b = connected_node.get_coordinate().to_real_point()
                vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
                vector_len = np.sqrt(np.power(vector[0], 2) + np.power(vector[1], 2))

                while current_distance <= vector_len:
                    ratio = current_distance / vector_len

                    # Move point_a inside the segment
                    point_a = (point_a[0] + (vector[0] * ratio), point_a[1] + (vector[1] * ratio))

                    # Insert point_a in positions
                    coordinate = Coordinate(x=point_a[0], y=point_a[1], map_origin=self._map_origin.get_x_y_tuple(), scale=self._scale)
                    positions.append(coordinate)

                    # Recalculate vector and its length and reset current distance
                    vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
                    vector_len = np.sqrt(np.power(vector[0], 2) + np.power(vector[1], 2))
                    current_distance = max_distance

                find_positions(connected_node, max_distance, current_distance - vector_len)

        for component in self._connected_components.values():
            starting_node = list(component)[0]
            visited_nodes = set()

            find_positions(starting_node, interval, interval)

        return positions