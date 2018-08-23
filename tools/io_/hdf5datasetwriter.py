import h5py
import os
import numpy as np


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1024, maxLabelLength=10, overwrite=False,
                 compression_level=9):
        # check if the parent path of the outputPath exists, if not, create it
        par_path = '/'.join(outputPath.split(os.path.sep)[:-1])
        if not os.path.exists(par_path):
            os.makedirs(par_path)

        # check if overwrite, if so, delete the old file.
        if overwrite:
            try:
                os.remove(outputPath)
            except:
                pass

        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already "
                             "exists and cannot be overwritten. Mannually delete "
                             "the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype=np.float, compression=compression_level)
        self.labels = self.db.create_dataset("labels", (dims[0], maxLabelLength), dtype=np.float,
                                             compression=compression_level)
        self.input_len = self.db.create_dataset("input_length", (dims[0],), dtype=np.int,
                                                compression=compression_level)
        self.label_len = self.db.create_dataset("label_length", (dims[0],), dtype=np.int,
                                                compression=compression_level)

        self.maxLabelLength = maxLabelLength

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": [],
                       "input_length": [], "label_length": []}
        self.idx = 0

    def add(self, rows, labels, input_length, label_length):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        self.buffer["input_length"].extend(input_length)
        self.buffer["label_length"].extend(label_length)

        # add the rows and labels to the buffer
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.input_len[self.idx:i] = self.buffer["input_length"]
        self.label_len[self.idx:i] = self.buffer["label_length"]
        self.idx = i
        self.buffer = {"data": [], "labels": [],
                       "input_length": [], "label_length": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=np.unicode)
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
