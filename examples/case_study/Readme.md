# FRED-MD Case Study
This case study shows a possible usage of the Error Correction Neural Network (ECNN) on a popular data set. First, we have to build the Python environment and download FRED-MD. Afterwards, we can run the benchmark and use the trained ECNN to create a heatmap forecast and a sensitivity analysis.


## Environment
Create environment: ``conda env create -f environment.yml``
Activate environment: ``conda activate prosper_nn``

Additionally install *tqdm*: ``pip install tqdm``


## Data
Download monthly data *2024-01.csv* from
https://files.stlouisfed.org/files/htdocs/fred-md/monthly/2024-01.csv
and place it in the current directory.

## Files
Short description of the files in the case study:
* ``config.py`` Stores the hyperparameters.
* ``fredmd.py`` Creates a PyTorch dataset with the option to select train, validation or test set. All timeseries are transformed to log-differences and rolling origins are created.
* ``models.py`` Defines the benchmark models. First a context vector is created either by an *Elman*, *GRU* or *LSTM* model. Afterwards, three different forecast approaches are implemented:
    * *Direct*: Maps the context vector to the forecast for all forecast horizons with an affine transformation.
    * *Recursive*: From the context vector, states for each step in the forecast horizon are created with the same recurrent model that is used to create the context vector. Finally, for each step in the forecast horizon, the state is mapped to the forecast with an affine linear transformation.
    * *Sequence to Sequence*: Similar approach as *Recursive*, but a second recurrent model of the same type is used to create the states in the forecast horizon.
* ``benchmark.py`` Creates dataset classes of FRED-MD for training, validation and testing. Afterwards all models are trained on the dataset and the trained ECNN is saved. The model state that performed best on the validation set is evaluated on the test set. The forecast performance is assessed individually for each forecast step.
* ``visualization.py`` Uses the saved ECNN to create a heatmap forecast and a sensitivity analysis.

## Replicate
1) Run the ``benchmark.py`` to train an ECNN ensemble and ten other benchmark models. The trained ECNN is saved.

2) Run ``visualization.py`` to use the saved ECNN to create a heatmap forecast and a sensitivity analysis.