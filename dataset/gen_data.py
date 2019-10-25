import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import utils.config as cfg

def gen_data():
    root = os.listdir(cfg.DATA_PATH)

    train, test, _, _ = train_test_split(root,
                                         range(0, len(root)),
                                         test_size=0.1,
                                         random_state=cfg.RANDOM_SEED)

    train, valid, _, _ = train_test_split(train,
                                         range(0, len(train)),
                                         test_size=0.1,
                                         random_state=cfg.RANDOM_SEED)
    set_dict = {
        'train': train,
        'valid': valid,
        'test' : test
    }

    dir_path = os.path.abspath(os.path.dirname(__file__))

    for name in ['train', 'valid', 'test']:
        with open(os.path.join(dir_path, f"{name}.txt"), "w") as file:
            print(f"generateing {name} dataset...")

            for filename in tqdm(set_dict[name]):
                out = str(os.path.join(cfg.DATA_PATH, filename))
                print(out, file=file)

if __name__ == '__main__':
    gen_data()
