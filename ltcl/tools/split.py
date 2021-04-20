import os
import glob
import random
VALIDATION_RATIO = 0.2

DIR = "/home/cmu_wyao/projects/data/"
if __name__ == "__main__":
    datum_names = glob.glob(os.path.join(DIR, "*.npz"))
    n_samples = len(datum_names)
    # Shuffle samples
    random.shuffle(datum_names)
    n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
    # Write training/val sample names to config files
    with open(os.path.join(DIR, "train.txt"), "w") as f:
        for datum_name in datum_names[:n_train_samples]:
            f.write('%s\n' % datum_name)
    with open(os.path.join(DIR, "val.txt"), "w") as f:
        for datum_name in datum_names[n_train_samples:]:
            f.write('%s\n' % datum_name)