import os
from stub_generator.stub_generator import StubGenerator

StubGenerator(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gibson_env_utilities', 'doors_dataset', 'door_sample.py'),
              ['DoorSample']).generate_stubs().write_to_file()