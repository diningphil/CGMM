from __future__ import absolute_import, division, print_function
from CGMMTF.TrainingUtilities import *
import sys

# Load hyper-params and other constants
from config import C, layers, use_statistics, max_epochs, threshold, batch_size

# ----------------------- Dataset creation  ----------------------- #
files = ['Images_Tasks/images/img1.png', 'Images_Tasks/images/img2.png']

def load_stats(filename):
    a = tf.data.Dataset.from_tensor_slices(
        tf.reshape(
            tf.decode_raw(tf.read_file(filename), out_type=tf.float64), [-1, A, C + 1]))
    return a

def create_dataset(files, batch_size):
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
    return dataset.batch(batch_size=batch_size)

batch_dataset = create_dataset(files, batch_size)


# Define a function that, given each image, constructs the statistics' tensor of shape (N, A, C)
def compute_statistics(inferred_states, file, A, C):
    print('TO BE IMPLEMENTED')
    # TODO it should not be memory intensive. Save portions of them as you go
    new_stats = np.full(shape=(len(inferred_states), A, C+1), fill_value=3.)

    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)

    if not os.path.exists(os.path.join(stats_folder, exp_name)):
        os.makedirs(os.path.join(stats_folder, exp_name))

    with open(os.path.join(stats_folder, exp_name) + '/' + file.split('/')[-1][:-4] + '_stats_' + str(0) + '.txt',
              'wb') as f:
        f.write(new_stats.tostring())


def merge_statistics(examples, L, C2):
    stats = None

    for l in range(0, L):
        example = examples[l]

        # Reshape image data into the original shape
        l_stats = tf.reshape(example, [1, A, C2])  # add dimension relative to L

        if stats is None:
            stats = l_stats
        else:
            stats = tf.concat([stats, l_stats], axis=0)

    return stats


#  --------------------------------------------------------------- #

# Define the model's params (which are given by the task at hand)
K = 255
A = 1

exp_name = 'prova'  # save the model with this name

# ------------------------------- Incrementally builds the network ------------------------------- #
variables_to_save = []
opts = tf.GPUOptions(allow_growth = True)
#with tf.Session(config=tf_config) as sess:
with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=opts)) as sess:
    print("LAYER 0")
    #'''
    mm = MultinomialMixture(C, K)
    mm.train(batch_dataset, sess, max_epochs=max_epochs, threshold=0., debug=False)

    # Add ops to save and restore the variables ('uses the variables' names')
    variables_to_save.extend([mm.prior, mm.emission])
    
    # For each file e.g. TFRecord
    for file in files:
        file_dataset = create_dataset([file], batch_size)

        print('INFERENCE...')
        # Returns ALL the inferred states in a numpy array (must fit in memory)
        inferred_states = mm.perform_inference(file_dataset, sess)

        print('STATISTICS...')
        # Compute the statistics and save them
        compute_statistics(inferred_states, file, A, C)


    #'''
    for layer in range(1, layers):
        # e.g 1 - [1, 3] = [0, -2] --> [0]
        # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        L = len(layer_wise_statistics)
        # Create the statistics dataset
        stats = []
        for previous_layer in layer_wise_statistics:
            stats_files = [
                os.path.join(stats_folder, exp_name) + '/' + file.split('/')[-1][:-4] + '_stats_' + str(previous_layer)
                + '.txt' for file in files]
            dataset = tf.data.Dataset.from_tensor_slices(stats_files)
            dataset = dataset.flat_map(lambda filename: load_stats(filename))
            stats.append(dataset)

        stats_dataset = tf.data.Dataset.zip(tuple(stats))
        stats_dataset = stats_dataset.map(lambda examples: merge_statistics(examples, L, C+1))
        batch_statistics = stats_dataset.batch(batch_size=batch_size)

        batch_dataset = batch_dataset.prefetch(batch_size)
        batch_statistics = batch_statistics.prefetch(batch_size)

        print('LAYER', layer)
        vs = VStructure(C, C, K, L, A, current_layer=layer)
        vs.train(batch_dataset, batch_statistics, sess, max_epochs=max_epochs, threshold=threshold)

        # Add ops to save and restore the variables ('uses the variables' names')
        variables_to_save.extend([vs.emission, vs.arcS, vs.layerS, vs.transition])

        for file in files:
            file_dataset = create_dataset([file], batch_size)

            print('INFERENCE...')
            # Returns ALL the inferred states in a numpy array (must fit in memory)
            inferred_states = vs.perform_inference(batch_dataset, batch_statistics, sess)

            print('STATISTICS...')
            # Compute the statistics
            compute_statistics(inferred_states[0], file, A, C)

        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(os.path.join(checkpoint_folder, exp_name)):
            os.makedirs(os.path.join(checkpoint_folder, exp_name))

        saver = tf.train.Saver(variables_to_save)
        print("Model saved in", saver.save(sess, os.path.join(checkpoint_folder, exp_name, 'model.ckpt')))


    # ------------------------------- Incremental inference ------------------------------- #

    incremental_inference(save_name, K, A, C, layers, use_statistics, target_dataset, adjacency_lists, sizes,
                          unigram_inference_name_train, statistics_inference_name, batch_size=batch_size)

