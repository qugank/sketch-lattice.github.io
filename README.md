# sketch-lattice

SketchLattice @ICCV21

This is the official implementation (PyTorch) of SketchLattice: Latticed Representation for Sketch Manipulation https://arxiv.org/abs/2108.11636

# Training Manual

## dataset
  
We used the <a href="https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset" target="_blank">QuickDraw Dataset<a>, which can be downloaded as per-class `.npz` files from <a href="https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn" target="_blank">Google Cloud<a>.
  
Some `.npz` examples in `./dataset` directory are available.

## training 

1. edit `generation_hyper_params.py`, setting `self.data_location` and `self.save_path`

2. running `python generation_sketch_gcn.py` for training

3. running `python generation_inference.py` for validating
  
  
Bibtex: 

    @inproceedings{yonggang2021sketchlattice,
        title={SketchLattice: Latticed Representation for Sketch Manipulation},
        author={Yonggang Qi, Guoyao Su, Pinaki Nath Chowdhury, Mingkang Li, Yi-Zhe Song},
        booktitle={ICCV},
        year={2021}
    }
