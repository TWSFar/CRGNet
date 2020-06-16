from datasets import visdrone, tt100k, dota, uavdt


def get_dataset(db_name, db_root):
    if db_name.lower() == 'visdrone':
        return visdrone.VisDrone(db_root)

    elif db_name.lower() == 'tt100k':
        return tt100k.TT100K(db_root)

    elif db_name.lower() == 'dota':
        return dota.DOTA(db_root)

    elif db_name.lower() == 'uavdt':
        return uavdt.UAVDT(db_root)

    else:
        raise NotImplementedError
