# Train model
python high_main.py --experiment neighbor --use_gpu --mode train --regularize --layer_size 16
# Get NNK neighbors for every layer and every channel
python high_main.py --experiment neighbor --mode plot --regularize --layer_size 16 --knn_layers 0,1,2,3,4,5,6,7,8,9,10,11 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --separate_relu
python high_main.py --experiment neighbor --mode plot --regularize --layer_size 16 --knn_layers 0,1,2,3,4,5 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# Plot activations when using def network_architecture_v2() (--separate_relu)
python high_main.py --experiment neighbor --mode activations --regularize --layer_size 16 --knn_layers 0,1,2,3,4,5,6,7,8,9,10,11 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --separate_relu
python high_main.py --experiment neighbor --mode activations --regularize --layer_size 16 --knn_layers 1,3,5,7,9,11 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --separate_relu
python high_main.py --experiment neighbor --mode activations --regularize --layer_size 16 --knn_layers 0,2,4,6,8,10 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --separate_relu
# Plot activations when using def network_architecture()
python high_main.py --experiment neighbor --mode activations --regularize --layer_size 16 --knn_layers 0,1,2,3,4,5 --knn_channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15