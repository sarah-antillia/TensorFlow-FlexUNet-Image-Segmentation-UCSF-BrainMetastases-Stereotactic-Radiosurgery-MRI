# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGenerator.py
# 2025/09/15 

import os
import shutil
import glob
import nibabel as nib
import numpy as np
import traceback
import cv2

class ImageMaskDatasetGenerator:

  def __init__(self, 
               input_dir = "", 
               output_dir= "", 
               angle     = cv2.ROTATE_90_CLOCKWISE, 
               resize    = 512):
    
    self.input_dir = input_dir 

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    if not os.path.exists(self.output_images_dir):
      os.makedirs(self.output_images_dir)

    if not os.path.exists(self.output_masks_dir):
      os.makedirs(self.output_masks_dir)

    self.angle = angle
    self.BASE_INDEX = 1000

    self.SEG_EXT    = "_seg.nii.gz"
    self.T1POST_EXT = "_t1post.nii.gz"
    self.RESIZE     = (resize, resize)
    self.file_format = ".png"
    
  def generate(self):
    subdirs = os.listdir(self.input_dir)
    subdirs = sorted(subdirs)
    for subdir in subdirs:
      subdir_fullpath = os.path.join(self.input_dir, subdir)
      print("=== subdir {}".format(subdir))
      seg_file    = subdir_fullpath + "/" + subdir + self.SEG_EXT
      t1post_file = subdir_fullpath + "/" + subdir + self.T1POST_EXT
      self.generate_mask_files(seg_file    ) 
      self.generate_image_files(t1post_file ) 
    
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image - min) / scale
    image = image.astype('uint8') 
    return image
    
  def generate_image_files(self, niigz_file):
    basename = os.path.basename(niigz_file) 
    nameonly = basename.replace(self.T1POST_EXT, "")
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape
  
    for i in range(d):
      img = fdata[:,:, i]
      filename  = nameonly + "_"+ str(i+self.BASE_INDEX) + self.file_format
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
        img   = self.normalize(img)
        img   = cv2.resize(img,self.RESIZE)
        img   = cv2.rotate(img, self.angle)
        cv2.imwrite(filepath, img)
        print("=== Saved {}".format(filepath))
     
  def generate_mask_files(self, niigz_file ):
    basename = os.path.basename(niigz_file) 
    nameonly = basename.replace(self.SEG_EXT, "")   

    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape

    for i in range(d):
      img = fdata[:,:, i]

      if img.any() >0:
        img   = cv2.resize(img,self.RESIZE)
        img = img*255.0
        img = img.astype('uint8')
 
        img = cv2.rotate(img, self.angle)
        filename  = nameonly + "_" + str(i+ self.BASE_INDEX) + self.file_format
        filepath  = os.path.join(self.output_masks_dir, filename)
        cv2.imwrite(filepath, img)
        print("--- Saved {}".format(filepath))


if __name__ == "__main__":
  try:
    input_dir  = "./UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN/"
    output_dir = "./UCSF-BrainMetastases-master/"

    if not os.path.exists(input_dir):
      raise Exception("Not found " + input_dir)   

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    angle = cv2.ROTATE_90_CLOCKWISE	

    generator = ImageMaskDatasetGenerator(input_dir = input_dir, 
                                            output_dir= output_dir, 
                                            angle  = angle)
    
    generator.generate()
      
  except:
    traceback.print_exc()

 