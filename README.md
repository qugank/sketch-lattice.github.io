# sketch-lattice

SketchLattice @ICCV21

# Training Manual

## dataset

dataset is available in the directory `fast_data_dT_0.5_rebuttal_iccv`

## training 

1. edit `generation_hyper_params.py`, setting `self.data_location` and `self.save_path`

2. running `python generation_sketch_gcn.py` for training

3. running `python generation_inference.py` for validating
  
