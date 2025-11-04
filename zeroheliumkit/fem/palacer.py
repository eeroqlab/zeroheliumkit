import os
import json
from dataclasses import dataclass, asdict, field
import subprocess


@dataclass
class ProblemConfig:
    Type: str = "Driven"
    Verbose: int = 2
    Output: str="postpro/"

    def __post_init__(self):
        if self.Type not in ["Eigenmode", "Driven", "Transient", "Electrostatic", "Magnetostatic"]:
            raise ValueError("Problem Type must be 'Eigenmode', 'Driven', 'Transient', 'Electrostatic', or 'Magnetostatic'.")

@dataclass
class ModelConfig:
    Refinement: dict
    Mesh: str = "mesh/meshfile.msh"
    L0: float = 1.0e-6

@dataclass
class MaterialsConfig:
    Attributes: list[int]
    Permeability: float = 1.0
    Permittivity: float = 1.0
    LossTan: float | list[float] = 0.0
    # Conductivity: float = 0.0
    # LondonDepth: float = 0.0

@dataclass
class PostProEnergyConfig:
    Index: int=1
    Attributes: list[int]=field(default_factory=lambda: [1])

@dataclass
class PostProProbeConfig:
    Index: int=1
    Center: list[float]=field(default_factory=lambda: [0,0,0])

@dataclass
class DomainConfig:
    Materials: list
    Postprocessing: dict

@dataclass
class ElementConfig:
    Attributes: list[int]
    Direction: str

@dataclass
class LumpedPortConfig:
    Index: int = 1
    R: float = 50.0
    Excitation: bool = True
    Elements: list[ElementConfig] = field(default_factory=list)

@dataclass
class BoundaryConfig:
    PEC: dict=field(default_factory=lambda: {"Attributes": [1]})
    #Absorbing: dict=field(default_factory=lambda: {"Attributes": [1], "Order": 1})
    LumpedPort: list[LumpedPortConfig]=field(default_factory=list)

@dataclass
class DrivenConfig:
    MinFreq: float=1        # GHz
    MaxFreq: float=10       # GHz
    FreqStep: float=0.1    # GHz
    Save: list=field(default_factory=lambda: [1,10])
    AdaptiveTol: float=1.0e-3

@dataclass
class SolverConfig:
    Order: int = 1
    Device: str = "CPU"
    Driven: DrivenConfig = field(default_factory=DrivenConfig)

@dataclass
class PalaceConfig:
    Problem: ProblemConfig
    Model: ModelConfig
    Domains: DomainConfig
    Boundaries: BoundaryConfig
    Solver: SolverConfig


class PalaceRunner:
    """
    A class to run Palace simulations.
    """

    def __init__(self, config: PalaceConfig, exec_path: str):
        self.config = config    
        self.exec_path = exec_path
        self.json_path = self.config.Problem.Output + 'palace.json'
        self.save_json(self.json_path)

    def save_json(self, path: str = 'palace.json'):
        """
        Save the Palace configuration to a JSON file.
        """
        with open(path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def run(self, multicore: int = None):
        command_to_cd = f'cd {os.getcwd()}'
        if multicore:
            command_to_run = self.exec_path + ' -np ' + str(multicore) + ' ' + self.json_path
        else:
            command_to_run = self.exec_path + ' ' + self.json_path
        command = command_to_cd + ' && ' + command_to_run
        # Use osascript to tell the Terminal application to open a new window and execute the command
        try:
            subprocess.run(['osascript', '-e', f'tell application "Terminal" to do script "{command}"'])
        except KeyboardInterrupt:
            message = 'Interrupted by user'
            print(message)