# sketch-lattice

SketchLattice @ICCV21

This is the official implementation (PyTorch) of SketchLattice: Latticed Representation for Sketch Manipulation https://arxiv.org/abs/2108.11636

# Training Manual

## dataset
  
We used the <a href="https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset" target="_blank">QuickDraw Dataset<a>, which can be downloaded as per-class `.npz` files from <a href="https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn" target="_blank">Google Cloud<a>.
  
Some `.npz` examples are available in `./dataset` directory.

## sketch to Graph and Adj

Before training and testing, you should run `python -u sketch2GraphAndAdjScript.py` for preprocessing datasets.
  
* Before script, you should edit `outPath`, `split_nums`, `node_nums` and `mode(train/test)`.
  
* After script, you can find `*_adjs_train(test).npz` and `*_nodes_train(test).npz` for training(testing) in the output directory.
  
## training 

1. edit `generation_hyper_params.py`, setting `self.data_location`, `self.save_path`,`self.category` and other parameters if you need.

2. running `python -u generation_sketch_gcn.py` for training.

## testing

The pre-trained models(encoder & decoder) and the corresponding parameters are available in `./models_32_150`.
  
1. edit `generation_hyper_params.py`, setting `self.data_location`, `self.save_path`,`self.category` and other parameters if you need.
  
2. running `python -u generation_inference.py` for validating.
  
## Bibtex: 
If you have some inspirations for your work, we would appreciate your quoting our paper.

    @inproceedings{yonggang2021sketchlattice,
        title={SketchLattice: Latticed Representation for Sketch Manipulation},
        author={Yonggang Qi, Guoyao Su, Pinaki Nath Chowdhury, Mingkang Li, Yi-Zhe Song},
        booktitle={ICCV},
        year={2021}
    }
