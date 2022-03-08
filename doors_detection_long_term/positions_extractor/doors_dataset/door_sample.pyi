#######################################################
# This stub file is automatically generated by stub-generator
# https://pypi.org/project/stub-generator/
########################################################

import os
import cv2.cv2
import numpy
import pandas
import generic_dataset.utilities.color
import typing
import generic_dataset.generic_sample
import generic_dataset.sample_generator
import generic_dataset.data_pipeline

def round(data, engine):
	...

def get_matterport_positive_colors():
	...

positive_colors: list

def create_pretty_semantic_image(self, color: generic_dataset.utilities.color.Color) -> DoorSample:
	"""
	Creates the pretty semantic image starting from semantic image.
	:param color: the color used to fill positive pixels
	:return:
	"""
	...

def is_positive(self, threshold: float):
	"""
	Changes the sample label according to the number of positive pixels.
	:param threshold:
	:return:
	"""
	...

def visualize(self) -> typing.NoReturn:
	"""
	This method visualizes the sample, showing all its fields. Remember to calculates all fields before calling this method.
	:return:
	"""
	...

def get_bboxes(self, threshold: float = 0.0) -> typing.List[typing.Tuple[int, int, int, int]]:
	"""
	Returns a list containing the bounding boxes calculated examining semantic image.
	:param threshold: parameter used to filter the bounding boxes that are too small
	:return:
	"""
	...

DOOR_LABELS: dict

class DoorSample(generic_dataset.generic_sample.GenericSample, metaclass=generic_dataset.sample_generator.MetaSample):
	def __init__(sample, label: int = 0):
		...
	def set_depth_image(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "depth_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to depth_image
		:return: the DoorSample instance
		"""
		...
	def get_depth_image(sample) -> numpy.ndarray:
		"""
		Returns "depth_image" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of depth_image
		"""
		...
	def create_pipeline_for_depth_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "depth_image".
		The pipeline is correctly configured, the data to elaborate are "depth_image"
		and the pipeline result is set to "depth_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "depth_image" and writes the result into "depth_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_depth_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of depth_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_pretty_semantic_image(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "pretty_semantic_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to pretty_semantic_image
		:return: the DoorSample instance
		"""
		...
	def get_pretty_semantic_image(sample) -> numpy.ndarray:
		"""
		Returns "pretty_semantic_image" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of pretty_semantic_image
		"""
		...
	def create_pipeline_for_pretty_semantic_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "pretty_semantic_image".
		The pipeline is correctly configured, the data to elaborate are "pretty_semantic_image"
		and the pipeline result is set to "pretty_semantic_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "pretty_semantic_image" and writes the result into "pretty_semantic_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_pretty_semantic_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of pretty_semantic_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_depth_data(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "depth_data" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to depth_data
		:return: the DoorSample instance
		"""
		...
	def get_depth_data(sample) -> numpy.ndarray:
		"""
		Returns "depth_data" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of depth_data
		"""
		...
	def create_pipeline_for_depth_data(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "depth_data".
		The pipeline is correctly configured, the data to elaborate are "depth_data"
		and the pipeline result is set to "depth_data".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "depth_data" and writes the result into "depth_data"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_depth_data(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of depth_data. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_bounding_boxes(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "bounding_boxes" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to bounding_boxes
		:return: the DoorSample instance
		"""
		...
	def get_bounding_boxes(sample) -> numpy.ndarray:
		"""
		Returns "bounding_boxes" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of bounding_boxes
		"""
		...
	def create_pipeline_for_bounding_boxes(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "bounding_boxes".
		The pipeline is correctly configured, the data to elaborate are "bounding_boxes"
		and the pipeline result is set to "bounding_boxes".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "bounding_boxes" and writes the result into "bounding_boxes"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_bounding_boxes(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of bounding_boxes. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_robot_pose(sample, value: dict) -> DoorSample:
		"""
		Sets "robot_pose" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to robot_pose
		:return: the DoorSample instance
		"""
		...
	def get_robot_pose(sample) -> dict:
		"""
		Returns "robot_pose" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of robot_pose
		"""
		...
	def set_positive_colors(sample, value: typing.List[typing.List[int]]) -> DoorSample:
		"""
		Sets "positive_colors" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to positive_colors
		:return: the DoorSample instance
		"""
		...
	def get_positive_colors(sample) -> typing.List[typing.List[int]]:
		"""
		Returns "positive_colors" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of positive_colors
		"""
		...
	def set_label(sample, value: int) -> DoorSample:
		"""
		Sets "label" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to label
		:return: the DoorSample instance
		"""
		...
	def get_label(sample) -> int:
		"""
		Returns "label" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of label
		"""
		...
	def set_semantic_image(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "semantic_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to semantic_image
		:return: the DoorSample instance
		"""
		...
	def get_semantic_image(sample) -> numpy.ndarray:
		"""
		Returns "semantic_image" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of semantic_image
		"""
		...
	def create_pipeline_for_semantic_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "semantic_image".
		The pipeline is correctly configured, the data to elaborate are "semantic_image"
		and the pipeline result is set to "semantic_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "semantic_image" and writes the result into "semantic_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_semantic_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of semantic_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_bgr_image(sample, value: numpy.ndarray) -> DoorSample:
		"""
		Sets "bgr_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raise FieldHasIncorrectTypeException if the given value has a wrong type
		:raise AnotherActivePipelineException if the field has an active pipeline (terminate it before setting a new value)
		:param value: the value to be assigned to bgr_image
		:return: the DoorSample instance
		"""
		...
	def get_bgr_image(sample) -> numpy.ndarray:
		"""
		Returns "bgr_image" value.
		If the field is an "numpy.ndarray" and it has an active pipeline, an exception is raised. Terminate it before obtain the fields value.
		:raises AnotherActivePipelineException if the field has an active pipeline
		:return: the value of bgr_image
		"""
		...
	def create_pipeline_for_bgr_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "bgr_image".
		The pipeline is correctly configured, the data to elaborate are "bgr_image"
		and the pipeline result is set to "bgr_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "bgr_image" and writes the result into "bgr_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_bgr_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of bgr_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def pipeline_fix_bgr_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "bgr_image".
		The pipeline is correctly configured, the data to elaborate are "bgr_image"
		and the pipeline result is set to "bgr_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "bgr_image" and writes the result into "bgr_image"
		:rtype: DataPipeline
		"""
		...
	def pipeline_depth_data_to_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "depth_data".
		The pipeline is correctly configured, the data to elaborate are "depth_data"
		and the pipeline result is set to "depth_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "depth_data" and writes the result into "depth_image"
		:rtype: DataPipeline
		"""
		...
	def pipeline_fix_semantic_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "semantic_image".
		The pipeline is correctly configured, the data to elaborate are "semantic_image"
		and the pipeline result is set to "semantic_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "semantic_image" and writes the result into "semantic_image"
		:rtype: DataPipeline
		"""
		...
	def create_pretty_semantic_image(self, color: generic_dataset.utilities.color.Color) -> DoorSample:
		"""
		Creates the pretty semantic image starting from semantic image.
		:param color: the color used to fill positive pixels
		:return:
		"""
		...
	def calculate_positiveness(self, threshold: float):
		"""
		Changes the sample label according to the number of positive pixels.
		:param threshold:
		:return:
		"""
		...
	def visualize(self) -> typing.NoReturn:
		"""
		This method visualizes the sample, showing all its fields. Remember to calculates all fields before calling this method.
		:return:
		"""
		...
	def get_bboxes_from_semantic_image(self, threshold: float = 0.0) -> typing.List[typing.Tuple[int, int, int, int]]:
		"""
		Returns a list containing the bounding boxes calculated examining semantic image.
		:param threshold: parameter used to filter the bounding boxes that are too small
		:return:
		"""
		...
	def save_field(sample, field_name: str, path: str, file_name: str) -> DoorSample:
		"""
		Saves the given field to disk in the specified path.
		:raise FieldDoesNotExistException if field_name does not correspond to a field name
		:raise FieldIsNotDatasetPart if the field doesn't belong to the dataset
		:raise AnotherActivePipelineException if there is an active pipeline for this field
		:raise FileNotFoundError if the path does not exist
		:param field_name: the name of the field to save
		:type field_name: str
		:param path: the path where to save the field value
		:type path: str
		:param file_name: the name of the file in which to save the field. The file extension is automatically added by the save function, so omit it in the name
		:type file_name: str
		:returns the sample instance
		:rtype: GenericSample
		"""
		...
	def load_field(sample, field_name: str, path: str, file_name: str) -> DoorSample:
		"""
		Loads the given field from disk saved in the given path.
		The field value is not returned by this method but it is automatically set to the sample class.
		To retrieve it, use the correspondent get method.
		:raise FieldDoesNotExistException if field_name does not correspond to a field name
		:raise FieldIsNotDatasetPart if the field doesn't belong to the dataset
		:raise AnotherActivePipelineException if there is an active pipeline for this field
		:raise FileNotFoundError if the path does not exist
		:param field_name: the name of the field to load
		:type field_name: str
		:param path: the path where loading the field's value
		:type path: str
		:param file_name: the name of the file in which the field is saved. The file extension is automatically added by the load function, so don't include it in the name
		:type file_name: str
		:return: the sample instance
		:rtype: GenericSample
		"""
		...
	def release_all_locks(sample) -> DoorSample:
		"""
		Releases all locks related to all fields of the sample instance
		:return: GenericSample instance
		:rtype: GenericSample
		"""
		...
	def acquire_all_locks(sample) -> DoorSample:
		"""
		Acquires all locks related to all fields of the sample instance
		:return: GenericSample instance
		:rtype: GenericSample
		"""
		...
	def __exit__(sample, exc_type, exc_value, exc_traceback):
		...
	def __enter__(sample) -> DoorSample:
		...
	...
