import numpy as np
from scipy.io import arff


def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='../../archives/UCR_UEA/Multivariate_arff/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TEST_DATA), np.array(TEST_LABEL)]
