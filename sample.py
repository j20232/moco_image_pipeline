from pathlib import Path

from pipeline import Kernel


if __name__ == "__main__":
    # Please change here
    competition_name = "bengaliai-cv19"
    index = "0001"
    input_path = Path(".").resolve() / "input"
    model_weight_path = Path(".").resolve() / "models" / competition_name / index / f"{index}.pth"
    cache_dir = None

    kernel = Kernel(competition_name, index,
                    input_path, model_weight_path, cache_dir=cache_dir)
    preds = kernel.predict()
