# sketch-lattice

SketchLattice @ICCV21
This is the official implementation (PyTorch) of SketchLattice: Latticed Representation for Sketch Manipulation https://arxiv.org/abs/2108.11636

# Training Manual

## dataset

dataset `fast_data_dT_0.5_rebuttal_iccv` is available is available <a href="https://drive.google.com/file/d/1fAbDodKgpRYHBcKisvF-M8dxbaEaSclh/view?usp=sharing" target="_blank">here<a>.

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
