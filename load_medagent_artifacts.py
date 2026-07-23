r"""
load_medagent_artifacts.py

Central loader for every model/object checkpointed in real_eda.ipynb,
so downstream scripts don't need to copy-paste the notebook's
"load next time" cells (cells 98 / 138 / 186).

Usage:
    from load_medagent_artifacts import load_los_artifacts, load_readmit_artifacts

    los = load_los_artifacts()
    los['model_v3'].predict(los['X_test'])

    readmit = load_readmit_artifacts()
    readmit['model_readmit'].predict(readmit['X_test'])

Note on a subtle bug this fixes: the notebook's load cells both dump
into globals() under the same names (X_train, X_test, y_train, y_test,
groups). Loading the readmission checkpoint AFTER the LOS checkpoint
silently overwrites the LOS versions of those variables. Loading into
two separate namespaced dicts (los['X_test'] vs readmit['X_test'])
avoids that collision entirely.

Path setup (do this once per machine):
    This script never hardcodes a drive letter or username, so it works
    the same way for Ivan, Alex, and Yichen -- each person just points
    it at wherever THEIR OWN checkpoints live, by setting an
    environment variable:

        # Windows (Command Prompt)
        set MEDAGENT_CHECKPOINT_DIR=E:\MedAgent\checkpoints

        # Windows (PowerShell)
        $env:MEDAGENT_CHECKPOINT_DIR = "E:\MedAgent\checkpoints"

        # Mac/Linux
        export MEDAGENT_CHECKPOINT_DIR=/Users/yourname/MedAgent/checkpoints

    If the env var isn't set, it falls back to a `checkpoints/` folder
    sitting next to this script -- so cloning the repo and dropping
    your own checkpoints in a `checkpoints/` folder next to this file
    also works with zero configuration.

    Important: the checkpoint .pkl files themselves are NOT in the repo
    (correctly -- MIMIC-derived data can't be redistributed under the
    Data Use Agreement). Each person has to generate their own by
    running the notebook's training cells against their own
    credentialed MIMIC-IV access. This script only saves everyone from
    re-writing the loading boilerplate -- it can't hand anyone data.
"""

import pickle
import os

# 1. Explicit env var, if set (works identically on any machine/OS)
# 2. Otherwise, a `checkpoints/` folder sitting next to this script
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
CHECKPOINT_DIR = os.environ.get('MEDAGENT_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)

# Matches the save list in cell 137 (LOS / regression pipeline)
LOS_OBJECT_NAMES = [
    'df', 'df_encoded', 'df_encoded_v2', 'df_encoded_v3', 'model_df',
    'model', 'model_v2', 'model_v3',
    'X_train', 'X_test', 'y_train', 'y_test', 'y_pred',
    'groups', 'dx_count', 'vitals_wide', 'labs_wide',
]

# Matches the save list in cell 185 (readmission / classification pipeline)
READMIT_OBJECT_NAMES = [
    'df_readmit', 'model_df_readmit', 'X_train', 'X_test', 'y_train', 'y_test',
    'X_train_under', 'y_train_under', 'groups', 'spw', 'model_readmit',
    'feature_columns', 'results_df',
]


def _load(names, suffix, save_dir):
    """Load a list of pickled object names from save_dir, each with the given filename suffix."""
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            f"Checkpoint folder not found: {save_dir}\n"
            f"Set MEDAGENT_CHECKPOINT_DIR to point at your own checkpoints, "
            f"or place them in a 'checkpoints/' folder next to this script. "
            f"See the module docstring for setup instructions."
        )
    loaded = {}
    for name in names:
        path = os.path.join(save_dir, f'{name}{suffix}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing checkpoint file: {path}\n"
                f"You likely need to run the training cells in real_eda.ipynb "
                f"yourself first to generate your own checkpoints -- these "
                f".pkl files aren't (and shouldn't be) stored in the repo."
            )
        with open(path, 'rb') as f:
            loaded[name] = pickle.load(f)
    return loaded


def load_los_artifacts(save_dir=CHECKPOINT_DIR):
    """Load every LOS regression object: model / model_v2 / model_v3, data splits, feature frames."""
    artifacts = _load(LOS_OBJECT_NAMES, suffix='', save_dir=save_dir)
    print(f'Loaded {len(artifacts)} LOS objects from {save_dir}')
    return artifacts


def load_readmit_artifacts(save_dir=CHECKPOINT_DIR):
    """Load every readmission classification object: model_readmit, data splits, feature_columns."""
    artifacts = _load(READMIT_OBJECT_NAMES, suffix='_readmit', save_dir=save_dir)
    print(f'Loaded {len(artifacts)} readmission objects from {save_dir}')
    return artifacts


def load_all(save_dir=CHECKPOINT_DIR):
    """Load both pipelines at once, namespaced under 'los' and 'readmit'."""
    return {
        'los': load_los_artifacts(save_dir),
        'readmit': load_readmit_artifacts(save_dir),
    }


if __name__ == '__main__':
    all_artifacts = load_all()
    print('\nLOS objects:', list(all_artifacts['los'].keys()))
    print('Readmission objects:', list(all_artifacts['readmit'].keys()))
