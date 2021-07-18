# SPS Mirror Descent


### 1. Install requirements

`pip install -r requirements.txt` 

### 2. Train and Validate

```python
python trainval.py -e mushrooms -sb ../results -d ../results -r 1
```

Argument Descriptions:
```
-e  [Experiment group to run like 'mushrooms' (the rest of the experiment groups are in exp_configs/sps_exps.py)] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-d  [Directory where the datasets are aved]
```

### 3. Visualize the Results

Follow these steps to visualize plots. Open `results.ipynb`, run the first cell to get a dashboard like in the gif below, click on the "plots" tab, then click on "Display plots". Parameters of the plots can be adjusted in the dashboard for custom visualizations.

<p align="center" width="100%">
<img width="65%" src="docs/vis.gif">
</p>


