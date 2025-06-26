from __future__ import print_function 
import sys
if sys.version_info[:2] < (3, 3): 
    old_print = print 
    def print(*args, **kwargs): 
        flush = kwargs.pop('flush', False) 
        old_print(*args, **kwargs) 
        if flush: 
            file = kwargs.get('file', sys.stdout) # Why might file=None? IDK, but it works for print(i, file=None) 
            file.flush() if file is not None else sys.stdout.flush()

# P1  import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys
import cv2
import csv
import time
import numpy as np
import pandas as pd
import datetime as dt
from keras import activations
from collections import Counter
from keras.models import load_model
from keras.preprocessing import sequence
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
script_start1 = dt.datetime.now()
np.set_printoptions(suppress=True)
tf.compat.v1.disable_eager_execution()

import keras.backend.tensorflow_backend as KTF
from keras.utils.generic_utils import CustomObjectScope
session_config = tf.ConfigProto(
      log_device_placement=True,
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      allow_soft_placement=True)


# , flush=True
print('Begin ...', flush=True)
# P2  define functions


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass

def name_seq(fasta_file):
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
    name_list, seq_list = [], []
    for line in lines:
        if '>' in line:
            name_list.append(line.strip().replace('>', ''))
            seq_list.append('')
        else:
            seq_list[-1] = '%s%s' % (seq_list[-1], line.strip().upper())
    return name_list, seq_list

def nucl2codon(nucl_seq):
    nucl_seq = nucl_seq.upper().replace('U', 'T')
    codon_num, codon_list, aa_list = int(len(nucl_seq) / 3), [], []
    codon_aa = {'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TGT': 'C', 'TGC': 'C',
                'GAC': 'D', 'GAT': 'D', 'GAG': 'E', 'GAA': 'E', 'TTC': 'F', 'TTT': 'F',
                'GGG': 'G', 'GGA': 'G', 'GGT': 'G', 'GGC': 'G', 'CAC': 'H', 'CAT': 'H',
                'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'AAG': 'K', 'AAA': 'K', 'CTA': 'L',
                'CTC': 'L', 'CTT': 'L', 'TTG': 'L', 'TTA': 'L', 'CTG': 'L', 'ATG': 'M',
                'AAT': 'N', 'AAC': 'N', 'CCC': 'P', 'CCT': 'P', 'CCA': 'P', 'CCG': 'P',
                'CAA': 'Q', 'CAG': 'Q', 'AGG': 'R', 'AGA': 'R', 'CGG': 'R', 'CGA': 'R',
                'CGT': 'R', 'CGC': 'R', 'AGT': 'S', 'TCG': 'S', 'TCC': 'S', 'TCT': 'S',
                'TCA': 'S', 'AGC': 'S', 'ACT': 'T', 'ACA': 'T', 'ACG': 'T', 'ACC': 'T',
                'GTC': 'V', 'GTA': 'V', 'GTT': 'V', 'GTG': 'V', 'TGG': 'W', 'TAC': 'Y', 'TAT': 'Y'}
    if len(nucl_seq[(codon_num - 1) * 3:]) != 3:
        print('*** Error! => Your sequence is not a cds!!!  Please check your sequence.', flush=True)
        sys.exit()
    else:
        for i in range(codon_num):
            codon_list.append(nucl_seq[0 + 3 * i: 3 + 3 * i])
            if codon_aa.get(nucl_seq[0 + 3 * i: 3 + 3 * i]) is not None:
                aa_list.append(codon_aa.get(nucl_seq[0 + 3 * i: 3 + 3 * i]))
            else:
                aa_list.append('*')
        if '*' in ''.join(aa_list[:-1]):
            print('## Note! => Stop codons or Unknown amino acids before the end of the sequence. ', flush=True)
        else:
            pass
    return codon_list

def coding_codon(codon_list):
    coding_list = []
    codonc4_coding = {'GCT': 1, 'GCC': 2, 'GCA': 3, 'GCG': 4, 'TGT': 5, 'TGC': 6, 'GAC': 7,
                      'GAT': 8, 'GAG': 9, 'GAA': 10, 'TTC': 11, 'TTT': 12, 'GGG': 13, 'GGA': 14,
                      'GGT': 15, 'GGC': 16, 'CAC': 17, 'CAT': 18, 'ATA': 19, 'ATC': 20, 'ATT': 21,
                      'AAG': 22, 'AAA': 23, 'CTA': 24, 'CTC': 25, 'CTT': 26, 'TTG': 27, 'TTA': 28,
                      'CTG': 29, 'ATG': 30, 'AAT': 31, 'AAC': 32, 'CCC': 33, 'CCT': 34, 'CCA': 35,
                      'CCG': 36, 'CAA': 37, 'CAG': 38, 'AGG': 39, 'AGA': 40, 'CGG': 41, 'CGA': 42,
                      'CGT': 43, 'CGC': 44, 'AGT': 45, 'TCG': 46, 'TCC': 47, 'TCT': 48, 'TCA': 49,
                      'AGC': 50, 'ACT': 51, 'ACA': 52, 'ACG': 53, 'ACC': 54, 'GTC': 55, 'GTA': 56,
                      'GTT': 57, 'GTG': 58, 'TGG': 59, 'TAC': 60, 'TAT': 61}
    for item in codon_list:
        if codonc4_coding.get(item) is None:
            coding_list.append('0')
        else:
            coding_list.append(str(codonc4_coding.get(item)))
    return coding_list

def coding_aa(aa_list):
    coding_list = []
    aac_coding = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
                  'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                  'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    for item in aa_list:
        if aac_coding.get(item) is None:
            coding_list.append('0')
        else:
            coding_list.append(str(aac_coding.get(item)))
    return coding_list

def count_ave_std(res_list, step):
    temp_ave, temp_std = [], []
    for s1 in range(step):
        temp_data = res_list[s1::step]
        temp_ave.append(np.average(temp_data))
        temp_std.append(np.std(temp_data))
    return temp_ave, temp_std


"""
~/anaconda3/envs/py27/bin/python CasMiner-Pred-V5.py ./In_seq/Q99ZW2-cas9.fa p80-raw
    # same to:
    ~/anaconda3/envs/py27/bin/python CasMiner-Pred-V5.py ./In_seq/Q99ZW2-cas9.fa p80-raw 0
~/anaconda3/envs/py27/bin/python CasMiner-Pred-V5.py ./In_seq/Q99ZW2-cas9.fa p80-raw 1
"""
# P3  setting parameters
infile = sys.argv[1]  # para 1/2
shuffle_p = sys.argv[2]  # p80, para 2/2
recoding = 0  # 1 => [True] or 0 => [Fasle]

if len(sys.argv[1:]) > 3:
    do_cam = int(sys.argv[3])  # 1 => [True] or 0 => [Fasle]
    recoding = int(sys.argv[4])  # 1 => [True] or 0 => [Fasle]
elif len(sys.argv[1:]) > 2:
    do_cam = int(sys.argv[3])  # 1 => [True] or 0 => [Fasle]
else:
    do_cam = 0

if recoding == 1:
    print("\t>>> Recoding the input sequence ...")

max_seq_len, fold, time_thd = 1820, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 100**100  # 180s
start_name = '.'.join(infile.split('/')[-1].split(".")[:-1])
model_dir = '/data1/xuguoshun/lab_work/CRISPR-Cas9/01_model_ana/MODELs/%s' % shuffle_p
all_pred_dir = "./Pred_res"
pred_dir = '%s/M%s_CasPred_%s' % (all_pred_dir, shuffle_p, start_name)  #  '_'.join(np.array(time.localtime(), dtype='str')[:3])
coding_file = '%s/pred_coding_file' % pred_dir
if not os.path.exists(all_pred_dir):
    os.makedirs(all_pred_dir)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
if not os.path.exists(coding_file):
    os.makedirs(coding_file)

if do_cam == 1:
    gc_dir = "%s/GradCam_res" % pred_dir
    csv_sir = "%s/csv_%s" % (gc_dir, start_name)
    pdf_dir = "%s/pdf_%s" % (gc_dir, start_name)
    if not os.path.exists(gc_dir):
        os.makedirs(gc_dir)
    if not os.path.exists(csv_sir):
        os.makedirs(csv_sir)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)

name_list, seq_list = name_seq(infile)

file_error_name = 'Error_in_M%s_%s.txt' % (shuffle_p, start_name)
seq_feature, file_coding_url = len(list(Counter(list(''.join(seq_list))))), " "

if seq_feature < 10:
    print("seq_feature: %s => Seq_type: Nucl" % seq_feature, flush=True)
    file_coding_url = "%s/%s_codonc4.txt" % (coding_file, start_name)
    if os.path.exists(file_coding_url) is False or recoding==1:
        file_coding = open(file_coding_url, 'w', 0)
        for name, seq in zip(name_list, seq_list):
            file_coding.write('0 %s ,%s ,\n' % (name.replace(' ', '-').replace(',', '*'), " ".join(coding_codon(nucl2codon(seq)))))
        file_coding.close()
        print('Created "%s"' % file_coding_url, flush=True)
    else:
        print('Existed "%s"' % file_coding_url, flush=True)
else:
    file_coding_url = "%s/%s_aac.txt" % (coding_file, start_name)
    if os.path.exists(file_coding_url) is False or recoding==1:
        file_coding = open(file_coding_url, 'w', 0)
        print("seq_feature: %s => Seq_type: Prot" % seq_feature, flush=True)
        for name, seq in zip(name_list, seq_list):
            if coding_aa(seq).count('0') > 100:
                """#########  E2: Ineffective amino acids #########"""
                print('!!! Error occur => Error 2: Ineffective amino acids !!!', flush=True)
                err_occur = open('%s/%s' % (pred_dir, file_error_name), 'a+', 0)
                err_occur.write('%s\n%s\nError 2: Please check your Amino acid sequence, there are many "gap" or "non-AA" letters.\n\n' % (name, seq))
                err_occur.close()
                # sys.exit()
            else:
                file_coding.write('Pred=> %s ,%s ,\n' % (name.replace(' ', '-').replace(',', '*'), " ".join(coding_aa(seq))))
        file_coding.close()
        print('Created "%s"' % file_coding_url, flush=True)
    else:
        print('Existed "%s"' % file_coding_url, flush=True)
time.sleep(1)

# P5  Loading coding file data
file_pred_name = 'M%s_pred_%s.res' % (shuffle_p, start_name)
file_pred_url = '%s/%s' % (pred_dir, file_pred_name)
file_pred = open(file_pred_url, 'w', 0)

print('Loading data...', flush=True)
if seq_feature < 10:
    print("Nucl-file: %s\tcodonc4-file: %s" % (infile, file_coding_url), flush=True)
else:
    print("Prot-file: %s\taac-file: %s" % (infile, file_coding_url), flush=True)
times1 = dt.datetime.now()
pred_data = pd.read_csv(file_coding_url, index_col=False, header=None)
print(pred_data.shape, flush=True)

y_test_ori = pred_data[0]
x_test_ori = pred_data[1]
name = y_test_ori
name = [item.split()[1] for item in name]
x_test = []
for pi in x_test_ori:
    nr = pi.split(' ')[0:-1]
    ndata = map(int, nr)
    x_test.append(ndata)
x_test = np.array(x_test)

times2 = dt.datetime.now()
print('Time spent: ' + str(times2 - times1), flush=True)

# P6  predict and record results
print('Being predicted by the model ...', flush=True)
pred_seq = sequence.pad_sequences(x_test, maxlen=max_seq_len, padding='post', truncating='post')
all_result0, all_result1, model_use,  = [], [], []

for f1 in fold:  # range(fold):
    script_start3 = dt.datetime.now()
    if (script_start3 - script_start1).total_seconds() > time_thd:
        """#########  E3: System is busy 1/3 #########"""
        print('!!! Error occur => Error 3-1: System is busy !!!', flush=True)
        err_occur = open('%s/%s' % (pred_dir, file_error_name), 'w', 0)
        print('Script Spent Time: ' + str(script_start3 - script_start1), flush=True)
        err_occur.write("Error 3-1: The system is busy and it is recommended to perform this step of analysis after filtering out the sequences.")
        err_occur.close()
        sys.exit()

    print('# === Fold: %s ===\n' % (f1 + 1), flush=True)
    model_url = '%s/%s' % (model_dir, 'Best_model_R%s_%s.h5' % ((f1 + 1), shuffle_p.replace("-raw", '')))

    KTF.clear_session()
    session = tf.Session(config=session_config)
    KTF.set_session(session)
    with CustomObjectScope({}):
        model = load_model(model_url)

    pred_result = model.predict(pred_seq)
    for i in range(len(pred_result)):
        all_result0.append(pred_result[i][0])
        all_result1.append(pred_result[i][1])
    # if f1 == 0:
    #     file_pred.write('*** Best_model_R%s_%s.h5 ***\n' % ((f1 + 1), shuffle_p.replace("-raw", '')))
    # else:
    #     file_pred.write('\n*** Best_model_R%s_%s.h5 ***\n' % ((f1 + 1), shuffle_p.replace("-raw", '')))
    # for p_name, pred_val in zip(name, pred_result):
    #     file_pred.write("%s\t%s\n" % (p_name, pred_val[1]))
    del model
ave0, std0 = count_ave_std(all_result0, len(name))
ave1, std1 = count_ave_std(all_result1, len(name))
# file_pred.write('# === Predict the probability of Cas9 protein ===\n# id\tAVE(Cas9 Yes)\tSTD(Cas9 Yes)\n')
file_pred.write('# id\tAVE(Cas9 Yes)\tSTD(Cas9 Yes)\n')
for i in range(len(name)):
    file_pred.write('%s\t%.6f\t%.6f\n' % (name[i], ave1[i], float(std1[i])))


if do_cam == 1:
    # """ ############### GRADcam ############### """
    class_sele = 1

    gene_id_xin, gene_seq_xin = name_seq(infile)
    dict_abb_num = {'A': '01', 'C': '02', 'D': '03', 'E': '04', 'F': '05', 'G': '06', 'H': '07',
                    'I': '08', 'K': '09', 'L': '10', 'M': '11', 'N': '12', 'P': '13', 'Q': '14',
                    'R': '15', 'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}
    gene_seq, gene_id = gene_seq_xin, gene_id_xin

    for set_seq in range(len(gene_seq)):
        script_start3 = dt.datetime.now()
        if (script_start3 - script_start1).total_seconds() > time_thd:
            """#########  E3: System is busy 2/3 #########"""
            print('!!! Error occur => Error 3-2: System is busy !!!', flush=True)
            err_occur = open('%s/%s' % (pred_dir, file_error_name), 'w', 0)
            print('Script Spent Time: ' + str(script_start3 - script_start1))
            err_occur.write("Error 3-2: The system is busy and it is recommended to perform this step of analysis after filtering out the sequences.", flush=True)
            err_occur.close()
            sys.exit()

        print('%s %s %s' % ('=' * 10, gene_id[set_seq], '=' * 10), flush=True)
        x_pred, pred_data = [], []
        for y in range(len(gene_seq[set_seq].strip())):
            if dict_abb_num.get(gene_seq[set_seq][y]) is not None:
                pred_data.append(dict_abb_num.get(gene_seq[set_seq][y]))
            else:
                pred_data.append(0)

        num_classes = 2
        pred_data = np.array(pred_data, dtype='int').tolist()
        x_pred.append(pred_data)
        x_pred = np.array(x_pred)
        x_pred = sequence.pad_sequences(x_pred, maxlen=max_seq_len, padding='post', truncating='post')
        heatmaps = np.zeros([1, max_seq_len], dtype=float)

        for model_num in fold:  # range(fold):
            script_start3 = dt.datetime.now()
            if (script_start3 - script_start1).total_seconds() > time_thd:
                """#########  E3: System is busy 3/3 #########"""
                print('!!! Error occur => Error 3-3: System is busy !!!', flush=True)
                err_occur = open('%s/%s' % (pred_dir, file_error_name), 'w', 0)
                print('Script Spent Time: ' + str(script_start3 - script_start1))
                err_occur.write("Error 3-3: The system is busy and it is recommended to perform this step of analysis after filtering out the sequences.", flush=True)
                err_occur.close()
                sys.exit()

            print("*** GradCam Fold: %s ***" % (model_num + 1), flush=True)
            model_url = '%s/%s' % (model_dir, 'Best_model_R%s_%s.h5' % ((model_num + 1), shuffle_p.replace("-raw", '')))

            KTF.clear_session()
            session = tf.Session(config=session_config)
            KTF.set_session(session)
            with CustomObjectScope({}):
                model = load_model(model_url)

            y_pred = model.predict(x_pred)
            ret = model.output[0, class_sele]  # set point
            last_conv_layer = model.get_layer("conv1d_2")
            fm = last_conv_layer.output
            grads = K.gradients(ret, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1))
            iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value,conv_layer_output_value = iterate([x_pred])
            for i in range(pooled_grads.shape[0]):
                conv_layer_output_value[:, i] *= pooled_grads_value[i]
            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = heatmap[np.newaxis, :]
            heatmaps = np.append(heatmaps, heatmap, axis=0)
        heatmap = heatmaps.mean(axis=0)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap[np.newaxis,:]
        heatmap=cv2.resize(heatmap, (max_seq_len, 1))
        heatmap_s=heatmap

        file_csv_name = "%s.csv" % (gene_id[set_seq])
        url_f = "%s/%s" % (csv_sir, file_csv_name)
        f = open(url_f, "w", 0)
        writer = csv.writer(f)
        writer.writerow(['#', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        for i in range(max_seq_len):
            if i+1 > len(gene_seq[set_seq]):
                print("early_stop_at: "+str(i), flush=True)
                break
            line = ['%s%s' % (i + 1, gene_seq[set_seq][i])]
            for s in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                if gene_seq[set_seq][i] == s:
                    line.append(heatmap[0][i])
                else:
                    line.append(0)
            writer.writerow(line)
        f.close()

        data = pd.read_csv(url_f).sum(axis=1)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 2.5))
        plt.rcParams["font.family"] = 'Arial'
        plt.plot(np.arange(len(data)) + 1, np.array(data), lw=1.5)
        plt.xlim(-32, len(data) + 32)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlabel('The index of Amino acids', fontsize=18)
        plt.ylabel('Probability', fontsize=18)
        plt.tight_layout()
        file_png_name = "%s.pdf" % (gene_id[set_seq])
        plt.savefig('%s/%s' % (pdf_dir, file_png_name), format='pdf', dpi=100)
        script_start4 = dt.datetime.now()
        print('Round Spent Time: ' + str(script_start4 - script_start3), flush=True)

print('End of script!!!', flush=True)
script_start5 = dt.datetime.now()
print('Script Spent Time: ' + str(script_start5 - script_start1), flush=True)
