# Dataset Preparation Instruction
- Raw data are located at ../../data/MoCap/Mocap_data
- Processed data are stored at ../../data/Mocap/Mocap_processed_data

## File Structure
## How to run 
- To convert Captury data into SMPL body data using blender
```python
blender -b -P "dataset_processing.py"
```
ouput data will be stored at the "output_dir" location in the script

- To process the labels
```python
python label_processing.py
```
ouput data will be stored at ../../data/MoCap/Mocap_processed_data/label.csv

- To add Temporal segmentation to processed csv files
```python
python temporal_segmentation.py
```