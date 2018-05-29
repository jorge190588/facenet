# Facenet project


# Install

# Run

## 1. Set environment variable to use facenet

Ubuntu
```
export PYTHONPATH=/notebooks/src
```

Windows
```
set PYTHONPATH=C:\Users\jorge\repository\facenet\files\src
```

## 2. Align data

### Original Example

Ubuntu 
```
for N in {1..4}; do python src/align/align_dataset_mtcnn.py lfw/raw lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
```
Note: remove for N in {1..4}; do and & done in the before code example


### Custom Example

Ubuntu
```
for N in {1..4}; do python src/align/align_dataset_mtcnn.py lfw/input lfw/output --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
```

Note: remove for N in {1..4}; do and & done in the before code example


## 2. Test Example

### Original Example

Ubuntu & Windows
```
python src/validate_on_lfw.py lfw/lfw_mtcnnpy_160 models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pair data/pairsOriginal.txt
```


### Custom Example

Ubuntu & Windows
```
python src/validate_on_lfw.py lfw/output models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pair data/pairs.txt  --lfw_batch_size 44
```

