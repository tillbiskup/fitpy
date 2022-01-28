import aspecd.utils


class_names = ['CalculatedDataset', 'CalculatedDatasetLHS']

for class_name in class_names:
    yaml = aspecd.utils.Yaml()
    ds = aspecd.utils.object_from_class_name(".".join(['fitpy.dataset',
                                                       class_name]))
    yaml.dict = ds.to_dict()
    yaml.serialise_numpy_arrays()
    yaml.write_to(".".join([class_name, 'yaml']))
