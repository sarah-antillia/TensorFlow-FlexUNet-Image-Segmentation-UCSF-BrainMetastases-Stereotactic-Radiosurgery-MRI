import os
import glob
import cv2
import shutil
import traceback

def jpg2png(input_dir, output_dir):
  files = glob.glob(input_dir + "/*.jpg")
  print(files)
  for file in files:
    image = cv2.imread(file)
    basename = os.path.basename(file)
    basename = basename.replace(".jpg", ".png")
    output_file = os.path.join(output_dir, basename)
    cv2.imwrite(output_file, image)
    print("Saved {}".format(output_file))

if __name__ == "__main__":
  try:
    input_dir = "./mini_test/images/"
    output_dir = "./png/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir)
    jpg2png(input_dir, images_dir)
    
    input_dir = "./mini_test/masks/"
    masks_dir   = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir)
    jpg2png(input_dir, masks_dir)


  except:
    traceback.print_exc()
