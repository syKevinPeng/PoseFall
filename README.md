# Generation of Novel Fall Animation with Configurable Attributes


## Training the CVAE
Before training, make sure you have the correct model type in the config file. Moreover, double-check the data path under the data-config.
``` bash
python -m src.train
```
## Generation
### Generate with 1E1D model
Modify in the config.yaml file the following parameters:
- generate_config - model_type: CVAE1E1D

Run the following command in the root directory of the project:
```bash
python -m src.generate_imgs
```

### Evaluate
```bash
python -m src.evaluate.evaluate
```

### Visualization
Before running the visulization script, which uses blender (make sure you have it installed), making sure that the input dir at the top of the script is set.

```bash
cd src/visualization
python process_outputs.py
```