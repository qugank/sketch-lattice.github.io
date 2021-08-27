# sketch-lattice

SketchLattice @ICCV21

# Training Manual

## dataset

dataset is available is available <a href="https://drive.google.com/file/d/1fAbDodKgpRYHBcKisvF-M8dxbaEaSclh/view?usp=sharing" target="_blank">here<a>.

## training 

1. edit `generation_hyper_params.py`, setting `self.data_location` and `self.save_path`

2. running `python generation_sketch_gcn.py` for training

3. running `python generation_inference.py` for validating
  
