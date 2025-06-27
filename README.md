# CasMiner

# Introduction
**CasMiner** is a deep learning tool for the high-throughput mining of Cas9 sequences and the rational design of optimized variants, capable of pinpointing key functional regions.

> Here, we provided:
>> (1) CasMiner(p80) model (**./01_CasMiner/p80/*.h5**);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2) Homology models (**./02_homology_models/**);  
>> (3) Training datasets (**./03_Model_training/**); &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(4) Model training logs (**./04_Model_logs/**);  
>> (5) Generalization ability (**./05_Model_Generalization/**);&nbsp;&nbsp;&nbsp;&nbsp;(6) Model prediction (**./06_Model_prediction/**).
  
And the `operating environment`, `model construction`, `powerful generalization capability`, and implementation with `**CasMiner** prediction` were presented.

![CasMiner](https://github.com/BRITian/CasMiner/blob/main/01_CasMiner/CasMiner.PNG)


# Operating environment
### System requirement
1. Python 2.7
2. tensorflow 1.15.0
3. keras 2.1.5
4. theano 1.0.5
5. opencv-python 4.1.2.30
6. matplotlib 2.2.5

### Quick Start to install the required program
1. Install the python 2.7 (from Anaconda https://www.anaconda.com/)
2. pip install tensorflow==1.15.0 (python=2.7)
3. pip install keras==2.1.5
4. pip install theano==1.0.5
5. pip install opencv-python==4.1.2.30
6. pip install matplotlib==2.2.5
7. git clone https://github.com/BRITian/CasMiner

# Model construction

Models (**CasMiner**) training were performed via `keras_unicas.py` (https://github.com/BRITian/CasMiner/blob/main/03_Model_training/keras_unicas.py).

### Script Configuration Guide
Before running the `keras_unicas.py` script, configure these critical parameters, and the data of **p80_trte**(https://github.com/BRITian/CasMiner/tree/main/03_Model_training/p80_trte) is provided here:

```python
# ===== Command Line Arguments =====
# Line 40: Set training data shuffle percentage (e.g., 80 for 80% of data)
percentage = sys.argv[1]  # Replace with integer value [10, 20, 30, 40, 50, 60, 70, 80(provided), 90, 100]

# Line 41: Set target fold number (0-9 for 10 folds)
tar_rep_id = sys.argv[2]  # Replace with integer fold number [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ===== Path Configuration =====
# Line 59: Training data directory path          # Format: p{percentage}_trte/ (e.g., p80_trte/)
mulu_train = "p%s_trte/" % percentage  # Modify path pattern if needed

# Line 60: Model save directory
mulu_models = "MODELs/"  # Default recommended, change for custom location

# Line 61: Training logs directory
mulu_logs = "LOGs/"  # Default recommended, change for custom location
```
### Running script
For example, here set `percentage = "80"`, `tar_rep_id = "5"` for the sequence internal rearrange 80% of the 5th fold data for model training, and the **modified script** is used to build the model with the following command line:  

	python keras_unicas.py 80 5  

# Generalization capability
We constructed four independent benchmark datasets:  
> UniRef100 Cas9 dataset (Cas9-100; 1,097 sequences);  
> Nuclease dataset (208 sequences);  
> Cas12 and Cas13 dataset (Cas12-13; 316 sequences);  
> Helicase dataset (570 sequences).  

Evaluation across these four completely independent test datasets demonstrates that **`CasMiner exhibits significantly stronger generalization capability`** compared to both homology-based models and alternative computational approaches.

![Model_Generalization](https://github.com/BRITian/CasMiner/blob/main/05_Model_Generalization/Model_Generalization.png)

# CasMiner prediction

Models (**CasMiner**) Prediction was performed via `CasMiner-Pred.py` (https://github.com/BRITian/CasMiner/blob/main/06_Model_prediction/CasMiner-Pred.py).
### Script Configuration Guide
Before running the `CasMiner-Pred.py` script, configure these critical parameters:
```python
# ===== Command Line Arguments =====
# Line 145: Input the sequence file to be predicted
infile = sys.argv[1]  # Input FASTA file (e.g., NH_Cas9_Cas12-13.fa, (https://github.com/BRITian/CasMiner/blob/main/06_Model_prediction/NH_Cas9_Cas12-13.fa))

# Line 146: Enter the name of the model
shuffle_p = sys.argv[2]  # model name [p10, p20, p30, p40, p50, p60, p70, p80(CasMiner), p90, p100]

# Line 150 and Line 153: (Optional) Whether to extract features or not
do_cam = int(sys.argv[3])  # Feature extraction? 0=False, 1=True (optional, default=0)

# Line 151: (Optional) Whether the input file needs to be re-encoded
recoding = int(sys.argv[4])  # Re-encode sequences? 0=False, 1=True (optional, default=0)

# ===== Path Configuration =====
# Line 162: Model save directory          # Format: p{shuffle_p}/ (e.g., p80/)
model_dir = '/data1/xuguoshun/lab_work/CRISPR-Cas9/01_model_ana/MODELs/%s' % shuffle_p  # !!! UPDATE THIS PATH

# Line 163: Prediction output directory
all_pred_dir = "./Pred_res"  # Default recommended, change for custom location
```

### Running script
Download the model folder (**01_CasMiner/p80/**(https://github.com/BRITian/CasMiner/tree/main/01_CasMiner/p80)), the predicted python file (**CasMiner-Pred.py**), and prepare the amino acid sequences file (**FILE_NAME.fa**, or fasta file with any extension) to be predicted, and then enter the python=2.7 environment to run:

**Condition 1**: (Large batch) sequences are only predicted without feature extraction and visualization **[do_cam=0(False)]**:

	python CasMiner-Pred.py FILE_NAME.fa p80  # python CasMiner-Pred.py NH_Cas9_Cas12-13.fa p80  

 or

 	python CasMiner-Pred.py FILE_NAME.fa p80 0  # python CasMiner-Pred.py NH_Cas9_Cas12-13.fa p80 0
  
**Condition 2**: (Small batch) sequence(s) is/are predicted, feature extraction and visualization **[do_cam=0(True)]**:

	python CasMiner-Pred.py FILE_NAME.fa p80 1  # python CasMiner-Pred.py NH_Cas9_Cas12-13.fa p80 1

**Condition 3**: Sequences needs to be re-encoded and predicted without feature extraction and visualization **[do_cam=0(False)]**:

	python CasMiner-Pred.py FILE_NAME.fa p80 0 1  # python CasMiner-Pred.py NH_Cas9_Cas12-13.fa p80 0 1

 
Simple sequence prediction, the prediction results will be saved in the "Mp80_pred_FILE_NAME.res" file, such as "Mp80_pred_NH_Cas9_Cas12-13.res"(https://github.com/BRITian/CasMiner/blob/main/06_Model_prediction/Mp80_CasPred_NH_Cas9_Cas12-13/Mp80_pred_NH_Cas9_Cas12-13.res).

Sequence features are extracted and all predictions will be saved in the "Mp80_CasPred_FILE_NAME" folder, such as "Mp80_CasPred_NH_Cas9_Cas12-13"(https://github.com/BRITian/CasMiner/tree/main/06_Model_prediction/Mp80_CasPred_NH_Cas9_Cas12-13).


### Result analysis 
In addition to the comment("#") rows, there are three columns. The first column is the IDs of the predicted sequences, the second column is the average value of Cas9-Yes probability (AVE) predicted by 10 models, and the third column is the average value (AVE) predicted by 10 models that the sequence is Standard deviation of probability of Cas9-Yes (STD) :

	# id	AVE(Cas9 Yes)	STD(Cas9 Yes)			# (comment row）
	Q99ZW2-Cas9	0.999605	0.000599
 	J7RUA5-Cas9	0.999492	0.000552
 	P08956-Nuclease	0.021309	0.019243
  	P38036-Helicase	0.025560	0.026849
	A0Q7Q2-Cas12a	0.018125	0.020241
	E4T0I2-Cas13a	0.025024	0.027344
 	...	...	...

As shown in the example (**Q99ZW2-Cas9**) results above, the larger the value in the second column (AVE) and the somaller the value in the third column (STD), the higher the probability that the sequence is Cas9.

![Model_Generalization](https://github.com/BRITian/CasMiner/blob/main/06_Model_prediction/Model_prediction.png)




