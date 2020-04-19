Based on ASPC-DA which acc(fmnist) = 0.591 acc(mnist) = 0.98
My IASPC-DA is acc(fmnist) = 0.62  acc(mnist) = 0.96
Install [Anaconda](https://www.anaconda.com/download/) with Python 3.6 version (_Optional_).   
Create a new env (_Optional_):   
```
conda create -n aspc python=3.6 -y   
source activate aspc  # Linux 
#  or 
conda activate aspc  # Windows
```
Install required packages:
```
pip install tensorflow-gpu==1.10 scikit-learn h5py  
```
### 2. Clone the code and prepare the datasets.

```
git clone https://github.com/BaronRay/IASPC-DA/
cd IASPC-DA
```

### 3. Run experiments.    

```bash
python run_exp.py
```
