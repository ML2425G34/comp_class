import os
import json
from abc import ABC, abstractmethod
import pandas as pd

class MLAnalytics(ABC):
    """
    A base class to handle shared functionality for analytics-related tasks
    like logging and signing pipeline configurations.
    """

    def __init__(self, base_dir):
        """
        Initializes common functionality for both signing and logging modes.
        
        Args:
            base_dir (str): The base directory that contains subdirectories for logs and results.
        """
        self.base_dir = base_dir
        self.run_log_dir = os.path.join(self.base_dir, "run_log")
        self.config_log_dir = os.path.join(self.base_dir, "config_log")
        self.csv_file_path = os.path.join(self.base_dir, "data_", "best_runs.csv")

        # Ensure necessary directories exist
        self.check_dir_exists(self.run_log_dir)
        self.check_dir_exists(self.config_log_dir)
        self.check_dir_exists(os.path.dirname(self.csv_file_path))

    def check_dir_exists(self, dir_path):
        """Ensures that the provided directory path exists. Creates it if necessary."""
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def get_current_id(self, target_dir):
        """
        Returns the current available ID by checking files in the specified directory.

        Args:
            target_dir (str): The directory to check for files with numeric IDs.
        
        Returns:
            int: The current maximum ID found in the directory.
        """
        if not target_dir or not os.path.exists(target_dir):
            return 0  # Default if the directory doesn't exist or is empty
        existing_files = os.listdir(target_dir)
        existing_ids = [
            int(f.split('_')[-1].replace('.json', '')) for f in existing_files if f.endswith('.json') and '_' in f
        ]
        return max(existing_ids, default=0)

    @abstractmethod
    def mode(self):
        """Return the mode in which the class is operating (e.g., 'signing' or 'logging')."""
        pass


class MLAuditer(MLAnalytics):
    """
    Subclass of MLAnalytics for signing and storing pipeline configurations.
    """

    @property
    def mode(self):
        return 'signing'

    def get_next_run_id(self):
        """
        Determines the next available run ID by checking the run log directory.
        
        Returns:
            int: The next available run ID.
        """
        current_run_id = self.get_current_id(self.config_log_dir)  # Use the shared method
        return current_run_id + 1

    def sign_and_save_config(self, pipeline_config):
        """
        Signs a pipeline configuration with a run_id and saves it as JSON.
        
        Args:
            pipeline_config (dict): The pipeline configuration to sign and save.
        
        Returns:
            str: Path to the saved JSON file.
        """
        signed_config = pipeline_config.copy()
        signed_config["config_id"] = self.get_next_run_id()  # Generate a run ID here

        json_file = os.path.join(self.config_log_dir, f"config_id_{signed_config['config_id']}.json")
        with open(json_file, "w") as f:
            json.dump(signed_config, f, indent=4)

        print(f"Signed pipeline configuration saved at: {json_file}")
        return json_file


class MLLogger(MLAnalytics):
    """
    Subclass of MLAnalytics for logging and storing results.
    """

    def __init__(self, base_dir, instructions_json):
        """
        Initializes the logger with directory and instructions.
        
        Args:
            base_dir (str): Base directory for saving results.
            instructions_json (dict): The instructions containing configuration details.
        """
        super().__init__(base_dir)
        self.config_id = instructions_json.get("config_id")  # Retrieve config_id from instructions

    @property
    def mode(self):
        return 'logging'

    def results_to_csv(self, result_data):
        """
        Appends results to a CSV file, injecting the config_id and run_id.
        
        Args:
            result_data (list of dicts): The result data to append to the CSV.
        
        Returns:
            None
        """
        if not result_data:
            raise ValueError("No result data to save.")

        # Convert the result data to a DataFrame
        df = pd.DataFrame(result_data)

        # Inject config_id and run_id into the DataFrame
        df['config_id'] = self.config_id
        df['run_id'] = self.get_current_id(self.run_log_dir)

        # Write to CSV (append mode)
        if not os.path.exists(self.csv_file_path):
            df.to_csv(self.csv_file_path, index=False)
        else:
            df.to_csv(self.csv_file_path, mode="a", header=False, index=False)

        print(f"Results saved to {self.csv_file_path}")
