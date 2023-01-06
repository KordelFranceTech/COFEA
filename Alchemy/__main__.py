# # main.py
# ########################################################################################################################
# # Main driver for training different types of neural nets over multimodal sensor data.
# ########################################################################################################################
#
# from Alchemy.neural_networks import convolutional_neural_network_tf as cnn_tf
# from Alchemy.neural_networks import neural_network_tf as nn_tf
# from Alchemy.neural_networks import convolutional_neural_network as cnn
# from Alchemy.neural_networks import neural_network as nn
# from Alchemy.neural_networks import pso_neural_network as pso_nn
# from Alchemy.data_processing import download_neural_net_data
# from Alchemy.data_processing import download_conv_neural_net_data
# from Alchemy.data_processing import download_pso_neural_net_data
#
#
# if __name__ == '__main__':
#
#     # TODO @kordelfrance: these filepaths are only for testing
#     # 1. regular feedforward neural network
#     # Replace with database paths for production
#     nn_filepath: str = './data/Sheet 1-AlchemyDataset_Sensor1_rev2.csv'
#     nn_data = download_neural_net_data(nn_filepath)
#     nn.build_neural_network(nn_data, False)
#     nn_tf.build_neural_network_tf(nn_data, False)
#
#     # TODO @kordelfrance: these filepaths are only for testing
#     # 2. convolultional neural network
#     # Replace with database paths for production
#     cnn_filepath: str = './data/AlchemyDataset_Sensors12.csv'
#     cnn_data = download_conv_neural_net_data(cnn_filepath)
#     cnn.build_convolutional_neural_network(cnn_filepath, False)
#     cnn_tf.build_convolutional_neural_network_tf(cnn_filepath, False)
#
#     # TODO @kordelfrance: these filepaths are only for testing
#     # 3. particle swarm optimization (PSO) neural network
#     # Replace with database paths for production
#     pso_filepath: str = './data/AlchemyDataset_Sensors12.csv'
#     pso_data = download_pso_neural_net_data(pso_filepath)
#     pso_nn.build_pso_neural_network(pso_filepath, False)
#
#
