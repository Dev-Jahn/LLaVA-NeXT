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
        cmd.append(f"--{key}")
        cmd.append(str(value))

    for key, value in config.get('optional_params', {}).items():
        if value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    return cmd


def main():
    args = parser.parse_args()
    config = load_config(args.config)
    ckpt_root = config['training_params'].pop('ckpt_root')
    config['training_params']['output_dir'] = os.path.join(ckpt_root, config['training_params']['run_name'])
    if 'model_path' in config['training_params']:
        config['training_params']['model_path'] = os.path.join(ckpt_root, config['training_params']['model_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_devices']
    os.environ['WANDB_PROJECT'] = config['wandb_project']
    cmd = build_command(config)
    print("Executing command:", ' '.join(cmd))
    # os.system('echo $CUDA_VISIBLE_DEVICES')
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
