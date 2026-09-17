import json
import os
from typing import Any, Dict


class ExperimentStorage:
    """
    Save and load RAGBench experiment results.
    """

    def __init__(self, output_dir: str = "data/experiments"):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    def save(
        self,
        result: Dict[str, Any],
    ) -> str:
        """
        Save an experiment result as JSON.

        Returns the path of the saved file.
        """

        experiment_name = result["experiment"]["name"]

        experiment_dir = os.path.join(
            self.output_dir,
            experiment_name,
        )

        os.makedirs(
            experiment_dir,
            exist_ok=True,
        )

        result_path = os.path.join(
            experiment_dir,
            "result.json",
        )

        with open(
            result_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return result_path

    def load(
        self,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """
        Load a previously saved experiment.
        """

        result_path = os.path.join(
            self.output_dir,
            experiment_name,
            "result.json",
        )

        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"Experiment result not found: {experiment_name}"
            )

        with open(
            result_path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    def load_all(self) -> list[Dict[str, Any]]:
            """
            Load all saved experiment results.

            Returns
            -------
            List of experiment result dictionaries.
            """

            experiments = []

            if not os.path.exists(self.output_dir):
                return experiments

            for experiment_name in sorted(
                os.listdir(self.output_dir)
            ):
                experiment_dir = os.path.join(
                    self.output_dir,
                    experiment_name,
                )

                if not os.path.isdir(experiment_dir):
                    continue

                result_path = os.path.join(
                    experiment_dir,
                    "result.json",
                )

                if not os.path.exists(result_path):
                    continue

                with open(
                    result_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    result = json.load(f)

                experiments.append(result)

            return experiments



                