from __future__ import absolute_import, division, print_function
from CGMMTF.TrainingUtilities import *

'''
def parse_image_files(filename_tensor):
    filename = tf.read_file(filename_tensor)
    image = tf.image.decode_png(filename, channels=1, dtype=tf.uint8)

    X = tf.reshape(image, [-1])

    # TODO change it with label vertex by vertex
    Y = X

    adjacency_lists = []
    arc_label = 0

    no_cols = image.shape[1]
    no_rows = image.shape[0]

    
    for i in range(0, no_rows):
        for j in range(0, no_cols):
            top_idx, bottom_idx, left_idx, right_idx = ((i - 1) % no_rows), ((i + 1) % no_rows), \
                                                       ((j - 1) % no_cols), ((j + 1) % no_cols)

            # Does not include a self-connection
            top_row = top_idx * no_cols
            bottom_row = bottom_idx * no_cols

            l = [(top_row + left_idx, arc_label), (top_row + j, arc_label), (top_row + right_idx, arc_label), \
                 (i * no_cols + left_idx, arc_label), (i * no_cols + right_idx, arc_label),
                 (bottom_row + left_idx, arc_label), (bottom_row + j, arc_label),
                 (bottom_row + right_idx, arc_label)]
            adjacency_lists.append(l)
    
    return X
'''

files = ['Images_Tasks/images/img1.png', 'Images_Tasks/images/img2.png']


# Take a dataset of files, create a 1-D tensor for each image, create a Dataset from such tensor, and then concatenate
# all datasets of these images
dataset = tf.data.Dataset.from_tensor_slices(files)
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.Dataset.from_tensor_slices(
            tf.reshape(
                tf.image.decode_png(tf.read_file(filename), channels=1, dtype=tf.uint8),
                [-1, 1])
        )
    ))

print(dataset)



# Hyper-Parameters
C = 10
K = 255
A = 1
# use_statistics = [1, 3]  # e.g use the layer-1 and layer-3 statistics
use_statistics = [1]
layers = 8  # How many layers you will train
max_epochs = 30

batch_size = 100000

# build minibatches from dataset
batch_dataset = dataset.batch(batch_size=batch_size)
print(batch_dataset)


# Training and inference phase
incremental_training(C, K, A, use_statistics, None, batch_dataset, layers, 'boh',
                         threshold=0, max_epochs=max_epochs, save_name='prove')


# Now recreate the dataset and the computation graph, because incremental_training resets the graph at the end
# (after saving the model)
target_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))

incremental_inference(save_name, K, A, C, layers, use_statistics, target_dataset, adjacency_lists, sizes,
                          unigram_inference_name_train, statistics_inference_name, batch_size=batch_size)

