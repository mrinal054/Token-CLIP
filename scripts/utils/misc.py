import os
import yaml

def save_config(config, save_dir, filename="config.yaml", verbose=False):
    """
    Save configuration to a YAML file.

    :param config: (Box or dict) Configuration object to save
    :param save_dir: (str) Directory path where the config file will be saved
    :param filename: (str) Name of the output config file (default: 'config.yaml')
    """
    # Convert Box to dict if needed
    if hasattr(config, "to_dict"):
        config_to_save = config.to_dict()
    else:
        config_to_save = dict(config)

    with open(os.path.join(save_dir, filename), "w") as f:
        yaml.safe_dump(config_to_save, f, default_flow_style=False)

    if verbose: print(f"Config saved to {os.path.join(save_dir, filename)}")
