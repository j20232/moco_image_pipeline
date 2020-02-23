from pathlib import Path

from pipeline import Kernel


if __name__ == "__main__":
    # Please change here
    competition_name = "bengaliai-cv19"
    index = "0001"
    input_path = Path(".").resolve() / "input"
    model_weight_path = Path(".").resolve() / "models" / competition_name / index / f"{index}.pth"
    config_path = Path(".").resolve() / "config" / competition_name / f"{index}.yaml"
    competition_yaml_path = Path(".").resolve() / "competition.yaml"
    output_path = input_path / competition_name / "output"

    kernel = Kernel(competition_name, input_path, config_path, competition_yaml_path,
                    model_weight_path, output_path)
    kernel.predict()
