# Basic step procedure prior to testsing the yolov8 model:

- Go to the UofA Spear shared folder/2025/2026/vision detection.
- Install the all the pictures for mallet, waterbottle and hammer.
- Install the labels for mallet, waterbottle and hammer

- place the images and labels into their obj_{object} folder in AUTO_NAV_MISSIN
into images and labels respectively

- run 'py whole_shebang.py' to train

- run 'test.py' to see results once the trainng is complete 

## Additional Features:
- change the souce variable argument in test.py to img_examples/{image name} which can be store in order to test images. Alternatively, use '0' for live camera
- change the 'epochs' number (higher the better) for desirable model 