# Generation of Novel Fall Animation with Configurable Attributes

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