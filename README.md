# pytorch-retinanet





3) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests

```

## Training

The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use



For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```

Note that the --csv_val argument is optional, in which case no validation will be performed.


For CSV Datasets (more info on those below), run the following script to validate:

`python csv_validation.py --csv_annotations_path path/to/annotations.csv --model_path path/to/model.pt --images_path path/to/images_dir --class_list_path path/to/class_list.csv   (optional) iou_threshold iou_thres (0<iou_thresh<1) `

It produces following resullts:

```
label_1 : (label_1_mAP)
Precision :  ...
Recall:  ...

label_2 : (label_2_mAP)
Precision :  ...
Recall:  ...
```

You can also configure csv_eval.py script to save the precision-recall curve on disk.

## creating csv annotations
retinanet_ann.py is the code to generate csv annotation files in the format described below. The arguments required are actual coco annotations path, train images directory path and validation images directory path. The file generates the required csv annotations files.

I divided total images in the dataset into two sets, train and validation and moved those images into their respective directories before generating csv annotation files.

## Visualization

To visualize the network detection, use `visualize.py`:

```
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```

## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/home/vijay_kumar/test/trainval/images/train/image_000000001.jpg,837,346,981,456,person
/home/vijay_kumar/test/trainval/images/train/image_000000002.jpg,215,312,279,391,car
/home/vijay_kumar/test/trainval/images/train/image_000000002.jpg,22,5,89,84,person
```

This defines a dataset with 3 images.
`image_000000001.jpg` contains a person.
`image_000000002.jpg` contains a car and a person.



### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
person,0
car,1
```

## Acknowledgements

- This git repo is cloned from (https://github.com/yhenon/pytorch-retinanet)
