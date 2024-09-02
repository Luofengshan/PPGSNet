# Code for《Progressive Prototype-Guided Segmentation Network for Building Extraction from High Resolution Remote Sensing Images》
# Dependencies
  * python==3.9.0
  * timm
  * catalyst==20.09
  * pytorch-lightning==1.5.9
  * albumentations==1.1.0
  * ttach
  * numpy
  * tqdm
  * opencv-python
  * scipy
  * matplotlib
  * einops
  * addict<br>
You also can run the following command to configure the required environment
```bash
pip install -r requirement.txt
```


# Train
To train the PPGSNet model, perform the following command:
```bash
python code/train.py -c config/whubuilding/model_v3.py
```
# Test
To test the PPGSNet model, perform the following command:
```bash
python code/test.py -c config/whubuilding/model_v3.py -o test_results/whubuilding/model_v3/ --rgb
```
