# Neural Point EM Fields (xxxx 2024)

<img src="demo/pred_cm.png">

### [Project Page](git@github.com:GeCao/neural-point-EM-field.git) 
#### Ge Cao, Zhen Peng


Neural Point EM Fields represent scenes with a light field living on a sparse point cloud. As neural volumetric 
rendering methods require dense sampling of the underlying functional scene representation, at hundreds of samples 
along with a ray cast through the volume, they are fundamentally limited to small scenes with the same objects 
projected to hundreds of training views. Promoting sparse point clouds to neural implicit light fields allows us to 
represent large scenes effectively with only a single implicit sampling operation per ray.

These point light fields are a function of the ray direction, and local point feature neighborhood, allowing us to 
interpolate the light field conditioned training images without dense object coverage and parallax. We assess the 
proposed method for novel view synthesis on large driving scenarios, where we synthesize realistic unseen views that 
existing implicit approaches fail to represent. We validate that Neural Point Light Fields make it possible to predict 
videos along unseen trajectories previously only feasible to generate by explicitly modeling the scene.

---

### Data Preparation
#### TODO:

---

### Requirements

Environment setup
```
conda create -n NeuralPointLF python=3.7
conda activate NeuralPointLF
```
Install required packages
```
conda install -c pytorch -c conda-forge pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install jupyterlab
pip install scikit-image matplotlib imageio plotly opencv-python
conda install pytorch3d -c pytorch3d
conda install -c open3d-admin -c conda-forge open3d
```

---
### Training and Validation
```
cd demo
python demo_EM.py
```

---
### Visualization of Results

TODO:

---
#### Citation
```
TODO:
```


