import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
CKPT_DIR = os.path.join(ROOT_DIR, 'saved_checkpoints')
LOG_DIR = os.path.join(ROOT_DIR, 'train_log')
WALK_LOG_DIR = os.path.join(ROOT_DIR, 'walk_log')


def ensure_dirs():
    for d in (CKPT_DIR, LOG_DIR, WALK_LOG_DIR):
        os.makedirs(d, exist_ok=True)
