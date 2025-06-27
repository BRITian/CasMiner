# CasMiner

Introduction
====
**CasMiner** is a deep learning tool for the high-throughput mining of Cas9 sequences and the rational design of optimized variants, capable of pinpointing key functional regions.

> Here, we provided:
>> (1) CasMiner(p80) model (**./01_CasMiner/p80/*.h5**);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2) Homology models (**./02_homology_models/**);  
>> (3) Training datasets (**./03_Model_training/**); &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(4) Model training logs (**./04_Model_logs/**);  
>> (5) Generalization ability (**./05_Model_Generalization/**);&nbsp;&nbsp;&nbsp;&nbsp;(6) Model prediction (**./06_Model_prediction/**);  
>> (7) Related images (**./07_Related_images/**).

![CasMiner](https://github.com/BRITian/CasMiner/blob/main/CasMiner.PNG)

System requirement
=====
1. Python 2.7
2. tensorflow 1.15.0
3. keras 2.1.5
4. theano 1.0.5
5. opencv-python 4.1.2.30
6. matplotlib 2.2.5

Quick Start to install the required program
=====
1. Install the python 2.7 (from Anaconda https://www.anaconda.com/)
2. pip install tensorflow==1.15.0 (python=2.7)
3. pip install keras==2.1.5
4. pip install theano==1.0.5
5. pip install opencv-python==4.1.2.30
6. pip install matplotlib==2.2.5
7. git clone https://github.com/BRITian/CasMiner

Predict the sequence 
====
Put the model folder (**CasMiner/MODELs/**), the predicted python file (**CasMiner-Pred.py**) and the amino acid sequences file (**FILE_NAME.fa**, or fasta file with any extension) to be predicted in the same directory, and then enter the python=2.7 environment to run:

**Condition 1**: (Large batch) sequences are only predicted without feature extraction and visualization **[pred_only=1(True)]**:

	python CasMiner-Pred.py FILE_NAME.fa 1
 
**Condition 2**: (Large batch) sequence(s) is/are predicted, feature extraction and visualization **[pred_only=0(False)]**:

	python CasMiner-Pred.py FILE_NAME.fa 0

The prediction result of the final model will be recorded in "Year_Month_Day_Cas9-Pred/Pred_p80_FILE_NAME.res"  **[pred_only=0** or **1]**

The feature extraction result of the final model will be recorded in "Year_Month_Day_Cas9-Pred/SEQ_NAME.csv"  **[pred_only=0]**

The prediction result of the final model will be recorded in "Year_Month_Day_Cas9-Pred/SEQ_NAME.png"  **[pred_only=0]**

Result analysis 
====
In addition to the comment("#") rows, there are three columns. The first column is the IDs of the predicted sequences, the second column is the average value of Cas9-Yes probability (AVE) predicted by 10 models, and the third column is the average value (AVE) predicted by 10 models that the sequence is Standard deviation of probability of Cas9-Yes (STD) :

	# === Predict the probability of Cas9 protein ===	# (comment row)
	# id	AVE(Cas9 Yes)	STD(Cas9 Yes)			# (comment row）
	Q99ZW2-Cas9	0.999692	0.000409

As shown in the example (**Q99ZW2-cas9.fa**) results above, the larger the value in the second column (AVE) and the somaller the value in the third column (STD), the higher the probability that the sequence is Cas9.. 
