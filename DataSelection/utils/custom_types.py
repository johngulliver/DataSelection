from enum import Enum, unique


@unique
class SelectorTypes(Enum):
    """
    Contains the names of the columns in the CSV file that is written by model testing.
    """
    PosteriorBasedSelector = "PosteriorBasedSelector"
    PosteriorBasedSelectorInv = "PosteriorBasedSelectorInv"
    TrainingLossDynamicsBasedSelector = "TrainingLossDynamicsBasedSelector"
    GraphBasedSelector = "GraphBasedSelector"
    DroppedCasesDynamicsBasedSelector = "DroppedCasesDynamicsBasedSelector"
