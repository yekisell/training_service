Training service project.
This project is used with [Rossmann store sales dataset](https://www.kaggle.com/c/rossmann-store-sales)

## Usage
***
1. Clone this repositore to your local machine.
2. Download [preprocessed data](https://drive.google.com/file/d/1cy9DV1zn2kOGy_pJjKZ83f3kse7XqgSm/view?usp=sharing)
3. Make sure that all packages from _requirements.txt_ are installed
4. Using terminal, enter:
```sh
python train.py
mlflow ui
```
If use want to change data path, size of validation dataset or parameters of the model, change configure parameters in _config/parameters.yaml_
