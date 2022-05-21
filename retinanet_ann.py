import json
import os
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

ann_path = '/home/vijay_kumar/test/trainval/annotations/bbox-annotations.json'
train_imgs_dir = '/home/vijay_kumar/test/trainval/images/train/'
valid_imgs_dir = '/home/vijay_kumar/test/trainval/images/val/'


def read_ann_json_file(ann_json_file_path):
  data = open(ann_json_file_path, 'r')
  data_dict = json.load(data)
  return data_dict
  
  


def get_image_annotations_data_as_df(ann_dict):  
  images_data = ann_dict['images']
  annotations_data = ann_dict['annotations']
  categories_data = ann_dict['categories']
  licenses_data = ann_dict['licenses']
  
  annotations_df = pd.DataFrame(annotations_data)
  images_df = pd.DataFrame(images_data)
  images_df.columns = ['file_name', 'width', 'height', 'image_id', 'license']
  image_annotations_df = pd.merge(annotations_df, images_df, on = 'image_id')
  return image_annotations_df, categories_data


def get_categories_dict(categories_data):
  categories_dict = {}
  for temp_dict in categories_data:
    categories_dict[temp_dict['id']] = temp_dict['name']
    
  return categories_dict 


def convert_xywh_to_xyxy(bboxes):
  return_bboxes = []
  for bbox in bboxes:
    x, y, w, h = bbox
    x2, y2 = x+w, y+h
    return_bboxes.append([x, y, x2, y2])
  
  return return_bboxes





def get_ann_csv_for_imgs(imgs_names, image_annotations_df, categories_dict, imgs_dir = None, imgs_type = 'train'):
  
  count = 0
  file_names_groups = image_annotations_df.groupby(['file_name'])
  all_lines = []
  for img_name in tqdm(imgs_names):
    #if count > 10:
      #break
    curr_file_name_group = file_names_groups.get_group(img_name)
    curr_bboxes = curr_file_name_group['bbox'].values.tolist()
    curr_categories_ids = curr_file_name_group['category_id'].values.tolist()
    curr_bboxes = convert_xywh_to_xyxy(curr_bboxes)
    #img = cv2.imread(os.path.join(imgs_dir, img_name))
    
    for index, bbox in enumerate(curr_bboxes):
      line = (os.path.join(imgs_dir, img_name), bbox[0], bbox[1], bbox[2], bbox[3], categories_dict[curr_categories_ids[index]])
      all_lines.append(line)
      #cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
    
      #Image.fromarray(img).save('./test' + str(count)+'.jpg')
    #count = count +1
  df = pd.DataFrame(np.array(all_lines), columns=['file_path', 'x1', 'y1', 'x2', 'y2', 'label'])
  df.to_csv('./'+imgs_type+'_ann.csv')
  #print(df.head())
  




def get_imgs_dir(image_annotations_df, categories_dict):
  train_imgs_names = os.listdir(train_imgs_dir)
  valid_imgs_names = os.listdir(valid_imgs_dir)
  get_ann_csv_for_imgs(train_imgs_names, image_annotations_df, categories_dict, train_imgs_dir, 'train')
  get_ann_csv_for_imgs(valid_imgs_names, image_annotations_df, categories_dict, valid_imgs_dir, 'valid')
  





def main():

  ann_dict = read_ann_json_file(ann_path)
  image_annotations_df, categories_data = get_image_annotations_data_as_df(ann_dict)
  categories_dict = get_categories_dict(categories_data)
  get_imgs_dir(image_annotations_df, categories_dict)
  
  
  
if __name__ == '__main__':
  main()