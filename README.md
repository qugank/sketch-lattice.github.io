SketchLattice: Latticed Representation for Sketch Manipulation
========================================================

SketchLattice @ICCV21

This is the official implementation (PyTorch) of SketchLattice: Latticed Representation for Sketch Manipulation https://arxiv.org/abs/2108.11636

<img src="./docs/front.png" width="650px"></img>

## Datasets and Preprocessing

### Datasets
  
There are 10 categories randomly selected from the <a href="https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset" target="_blank">QuickDraw Dataset<a> for all experiments. You can dowanload the data (one `.npz` file per class) from <a href="https://drive.google.com/file/d/1spj0eHU8HPtp1ET-3FVjWsja2G8F8CSF/view?usp=sharing" target="_blank">Google Cloud<a>.

After downloading, please unzip and place all the npz files into the `./dataset` directory.

### Sketch to Graph and Adj

To get started, a preprocess step needs to be done firstly by using the script `sketch2GraphAndAdjScript.py`. You can simply run the following command.
  ```python
    python -u sketch2GraphAndAdjScript.py
  ```
  
1. Before running the script, you should edit the following haperparameters:
  * `outPath`: Path to place the preprocessed datasets.
  * `split_nums`: Sampling density or Grid n, the default value is 32.
  * `node_nums`: Graph Nodes v, the default value is 150.
  * `mode(train/test)`: Preprocess on the train/test datasets.
  
2. After running the script, you will get `*_adjs_train(test).npz` and `*_nodes_train(test).npz` for training(testing) in the output directory.

## Training and Testing
  
### Setup
  
Setup environment via requirements.txt

```bash
  pip install -r requirements.txt
```
  
### Train

1. Before running the script, you should edit `generation_hyper_params.py` to modify the following haperparameters if you need:
  * `self.data_location`: Path to place the preprocessed datasets.
  * `self.save_path`: Path to place checkpoints and results.
  * `self.category`: Categories you choose to train or validate.
  * `self.row_column`: Sampling density or Grid n, the default value is 32.
  * `self.graph_number`: Graph Nodes v, the default value is 150.
  * `self.mask_prob`: Corruption levels p, the default value is 0.1.

2. For training, run
  ```python
    python -u generation_sketch_gcn.py
  ``` 

### Test (Reconstruct Sketches)

Trained models (encoder & decoder) are available in `./models_32_150`.
  
1. Before running the testing script, you should edit `generation_hyper_params.py` to modify the haperparameters as well.

2. For validating, run
  ```python
    python -u generation_inference.py
  ``` 
  
## Bibtex: 
Thank you for citing our work if it is helpful!

    @inproceedings{yonggang2021sketchlattice,
        title={SketchLattice: Latticed Representation for Sketch Manipulation},
        author={Yonggang Qi, Guoyao Su, Pinaki Nath Chowdhury, Mingkang Li, Yi-Zhe Song},
        booktitle={ICCV},
        year={2021}
    }
