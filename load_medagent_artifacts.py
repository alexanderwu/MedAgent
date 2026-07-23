"""
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
"""

import pickle
import os

CHECKPOINT_DIR = r'E:\MedAgent\checkpoints'

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
    loaded = {}
    for name in names:
        path = os.path.join(save_dir, f'{name}{suffix}.pkl')
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
