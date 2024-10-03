import os
import subprocess
import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def build_command(config):
    cmd = config['base_command']

    for key, value in config['training_params'].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    for key, value in config.get('optional_params', {}).items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

    return cmd


def main():
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_devices']
    cmd = build_command(config)
    print("Executing command:", ' '.join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
