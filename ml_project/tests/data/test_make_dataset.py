from heart_disease_classifier.data.make_dataset import read_data, write_data, split_train_val_data
from heart_disease_classifier.entities import SplittingParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_write_dataset(dataset_path: str, out_path: str):
    tmpdata = read_data(dataset_path)
    tmp_path = write_data(tmpdata, out_path)
    assert tmp_path == out_path


def test_split_dataset(tmpdir, dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=31337, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10