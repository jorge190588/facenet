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

Ubuntu
```
for N in {1..4}; do python src/align/align_dataset_mtcnn.py lfw/input lfw/output --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
```

Note: remove for N in {1..4}; do and & done in the before code example


## 2. Test Example

Ubuntu & Windows
```
python src/validate_on_lfw.py lfw/output models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pair data/pairs190.txt  --lfw_batch_size 236
```

## 3. Train a classifier on LFW (Custom Example)

### Train
```
python src/classifier.py TRAIN lfw/output models/20180402-114759/20180402-114759.pb models/20180402-114759/lfw_classifier10x800.pkl --batch_size 100 --min_nrof_images_per_class 10 --nrof_train_images_per_class 800 --use_split_dataset
```

### Classifier
```
python src/classifier.py CLASSIFY lfw/output models/20180402-114759/20180402-114759.pb models/20180402-114759/lfw_classifier.pkl --batch_size 236 --min_nrof_images_per_class 700 --nrof_train_images_per_class 300 --use_split_dataset
```


## Command Tools

1. xcopy in windows
```
xcopy /S g:\output\Camila_Santos\*.png Camila_Santos\
```