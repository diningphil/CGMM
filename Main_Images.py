from __future__ import absolute_import, division, print_function
from CGMMTF.TrainingUtilities import *

# Load hyperparams and other constants
from config import C, layers, use_statistics, max_epochs, threshold, batch_size

# ----------------------- Dataset creation  ----------------------- #
files = ['Images_Tasks/images/img1.png', 'Images_Tasks/images/img2.png']

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
    return np.ones(shape=(len(inferred_states), A, C+1))

#  --------------------------------------------------------------- #

# Define the model's params (which are given by the task at hand)
K = 255
A = 1

exp_name = 'prova'  # save the model with this name

# ------------------------------- Incrementally builds the network ------------------------------- #
variables_to_save = []
with tf.Session() as sess:
    print("LAYER 0")

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
        # Compute the statistics
        new_stats = compute_statistics(inferred_states, file, A, C)

        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)

        if not os.path.exists(os.path.join(stats_folder, exp_name)):
            os.makedirs(os.path.join(stats_folder, exp_name))

        np.save(os.path.join(stats_folder, exp_name) + '/' + file.split('/')[-1] + '_stats_' + str(0), new_stats)

    for layer in range(1, layers):
        print("LAYER", layer)

        # e.g 1 - [1, 3] = [0, -2] --> [0]
        # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        L = len(layer_wise_statistics)

        # print(layer_wise_statistics)


        #RISOLVI QUESTO E POI PENSI AL RESTO

        # TODO problema con la batch dimension. credo di averlo gia' risolto questo
        stats_dataset = recover_statisticsNEW(exp_name, layer_wise_statistics, A, C)
        batch_statistics = stats_dataset #stats_dataset.batch(batch_size=batch_size)


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
            new_stats = compute_statistics(inferred_states, file, A, C)

            np.save(os.path.join(stats_folder, exp_name) + '/' + file.split('/')[-1] + '_stats_' + str(layer), new_stats)

        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(os.path.join(checkpoint_folder, exp_name)):
            os.makedirs(os.path.join(checkpoint_folder, exp_name))

        saver = tf.train.Saver(variables_to_save)
        print("Model saved in", saver.save(sess, os.path.join(checkpoint_folder, exp_name, 'model.ckpt')))


    # ------------------------------- Incremental inference ------------------------------- #

    incremental_inference(save_name, K, A, C, layers, use_statistics, target_dataset, adjacency_lists, sizes,
                          unigram_inference_name_train, statistics_inference_name, batch_size=batch_size)

