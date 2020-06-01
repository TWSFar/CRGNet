from datasets import visdrone, tt100k, dota15, dotaall


def get_dataset(db_name, db_root):
    if db_name.lower() == 'visdrone':
        return visdrone.VisDrone(db_root)

    elif db_name.lower() == 'tt100k':
        return tt100k.TT100K(db_root)

    elif db_name.lower() == 'dota15':
        return dota15.DOTA15(db_root)

    elif db_name.lower() == 'dotaall':
        return dotaall.DOTAAll(db_root)

    else:
        raise NotImplementedError
