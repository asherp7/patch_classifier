import numpy as np
import h5py

# create arrays:
d1 = np.random.random(size=(1000, 20))
d2 = np.random.random(size=(1000, 200))
print(d1.shape, d2.shape)
#  initialize h5 file in write mode:
hf = h5py.File('data.h5', 'w')

hf.create_dataset('dataset_1', data=d1)
hf.create_dataset('dataset_2', data=d2)

# write to disk:
hf.close()

# open file in read mode:
hf = h5py.File('data.h5', 'r')

print("keys:", hf.keys())

# get dataset:
n1 = hf.get('dataset_1')
print(type(n1))
# convert to numpy:
n1 = np.array(n1)
print(type(n1))

hf.close()

def create_data_set2(f_name, data_set_names, time, dt=0.025, num_samples=100, window_size=100, num_syn=100,
                     win_ratio=0.25):
    data_name, labels_name = data_set_names
    file = h5py.File(f_name, 'w')
    file.create_dataset(name=data_name, shape=(0, num_syn * window_size), dtype=np.uint8, maxshape=(None, None),
                        chunks=True)
    file.create_dataset(name=labels_name, shape=(0,), dtype=np.uint8, maxshape=(None,))

    file.close()

    with h5py.File(f_name, 'a') as db:
        chunk_size = num_samples // 100
        for i in range(100):

            samples_counter = 0
            while samples_counter < chunk_size:
                windows, labels = create_windows2(time, window_size=window_size, num_syn=num_syn, ratio=win_ratio)
                if windows is not None and labels is not None:
                    samples_counter += len(windows)

                    db[data_name].resize(db[data_name].shape[0] + windows.shape[0], axis=0)
                    db[data_name][-windows.shape[0]:, :] = windows

                    db[labels_name].resize(db[labels_name].shape[0] + len(labels), axis=0)
                    db[labels_name][-windows.shape[0]:] = labels
                    print("samples: {0}".format(samples_counter))
            print("i:{0}".format(i))

