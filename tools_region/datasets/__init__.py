from datasets import visdrone


def get_dataset(db_name, db_root):
    if db_name in ['visdrone', 'Visdrone', 'VisDrone']:
        return visdrone.VisDrone(db_root)
    else:
        raise NotImplementedError
