from dataclasses import dataclass, field

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the training pipeline."""
    test_size: float = 0.25
    random_state: int = 44
    n_estimators: int = 100
    max_depth: int | None = None
    data_path: str = "./data/penguins.csv" 
    target: str = "species"
    num_columns : list = field(default_factory=lambda: ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    cat_columns: list = field(default_factory=lambda: ['island', 'sex'])