import os
import shutil

def clear(keep=None):
    if keep == None:
        for f in [x for x in os.listdir("DATA") if not os.path.isdir(os.path.join("DATA",x)) and x not in ["ELEMENTS.py", "feature_labels.py"]]:
            os.remove(os.path.join("DATA", f))

    else:
        for f in [x for x in os.listdir("DATA") if not os.path.isdir(os.path.join("DATA",x)) and x not in keep and x not in ["ELEMENTS.py", "feature_labels.py"]]:
            os.remove(os.path.join("DATA", f))

    return

if __name__ == "__main__":
    clear(keep=["phase_field_dataset.csv"])
