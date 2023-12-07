import mne
import pandas as pd
import numpy as np
import scipy.io
mne.set_log_level('CRITICAL')

def all_chs():
    mne.set_log_level('CRITICAL')
    raw = mne.io.read_epochs_eeglab('/home3/mci-eeg/data/target_lock/s1003.set')
    chs = [hey['ch_name'] for hey in raw.info['chs']]
    chs = {chs[i]:i for i in range(len(chs))}
    return chs

def TT(tmin = -2, tmax = 4):
    path = '/home3/mci-eeg-sep6-2023/s45/eeg'
    epochs = mne.io.read_epochs_eeglab(f'{path}/sbj4521flanker_sess01_artifactfree.set')
    epochs = epochs.crop(tmin = tmin, tmax = tmax)
    return epochs.times

def convert_isi (isi):
    return int(isi*512)/512

def subjectIDs():
    subjectID_list = {}
    subjectID_list['Healthy']  = ['1131','1167','1189','1243','1260','1261','1353','1357','1358','1367','1368','1374','1375','1376','1379','1380','1382','1383','1384','1385','1386','1387','1388','1391','1392','1393','1395','1396','1397','1398','1400','1402','1403','1405','1406','1407','1409','1602','2172','3018','4501','4503','4504','4505','4510','4511','4513','4514','4515','4518','4519','4521','4523','4524','4531','4532','4533','4534','4537','4541']
    subjectID_list['MCI'] = ['4502','4506','4507','4508','4509','4512','4516','4517','4520','4522','4526','4527','4529','4530','4535','4536','4538','4539','4544','4545','4548','4549']
    return subjectID_list

def df_subject_query (subjectID):
    params = ['cueloc','salientloc','coninc','RT','hit','isi']
    if subjectID[0] == '4':
        dire = 's45'
    else:
        dire = 's123'
    path = f'/home3/mci-eeg-sep6-2023/{dire}'
    d = {param:[] for param in params}
    d['index'] = list(range(576))
    for block_id in range(1,16+1):
        mat = scipy.io.loadmat(f'{path}/mat/flanker_allages_EEG_sbj{subjectID}_session1_block{block_id}.mat')
        for param in params:
            d[param].extend(mat['p'][param][0][0][0])

    matR = scipy.io.loadmat(f'{path}/threshold/sbj{subjectID}flanker_sess01_rejthreshold.mat')
    d['reject'] = matR['reject'][0]
    d['cond'] = []
    d['cue-di'] = []
    for i in range(576):
        cueloc = d['cueloc'][i]
        saloc = d['salientloc'][i]
        coninc = d['coninc'][i]
        if cueloc in [11,12,1,2,3]:
            di = 'R'
        elif cueloc in [5,6,7,8,9]:
            di = 'L'
        else:
            di = 'M'
        d['cue-di'].append(di)

        if saloc == 0 and di != 'M':
            cond = f'{di}NS{coninc}'
        elif saloc != cueloc and di != 'M':
            cond = f'{di}SA{coninc}'
        else:
            cond = 'X'
        d['cond'].append(cond)
    del mat
    return pd.DataFrame(d)



def subject_query (subjectID,cue_tar,csd=True,thresh=1):
    mne.set_log_level('CRITICAL')
    target_tmin, target_tmax = -2,4
    cue_tmin, cue_tmax = -2,4
    if cue_tar == 'cue_lock':
        baseline_window = (-0.35,-0.15)
    elif cue_tar == 'target_lock':
        baseline_window = (-0.2,0)
        
    if subjectID[0] == '4':
        dire = 's45'
    else:
        dire = 's123'
        
    path = f'/home3/mci-eeg-sep6-2023/{dire}/eeg'
    epochs = mne.io.read_epochs_eeglab(f'{path}/sbj{subjectID}flanker_sess01_artifactfree.set')
    epochs = epochs.pick_types(eeg=True)
    df = df_subject_query(subjectID)
    
    if cue_tar == 'cue_lock':
        epochs = epochs.apply_baseline(baseline_window)
        epochs = epochs.crop(tmin = cue_tmin, tmax = cue_tmax)
    elif cue_tar == 'target_lock':
        new_epochs = []
        for i in range(576):
            isi = df[df['index'] == i].isi.values[0]
            isi = convert_isi(isi)
            new_epoch = epochs[i].shift_time(-isi, relative=True).crop(tmin = target_tmin, tmax = target_tmax)
            new_epochs.append(new_epoch)
        epochs = mne.concatenate_epochs(new_epochs)
        epochs = epochs.apply_baseline(baseline_window)
    else:
        print("ERROR")
        return
    
    if csd:
        epochs = epochs.set_eeg_reference(projection=True).apply_proj()
    
    epochs_np = epochs._data
    conds = ['LNS1','LNS2','RNS1','RNS2','LSA1','LSA2','RSA1','RSA2','NS1','NS2','SA1','SA2']
    df = df_subject_query(subjectID)
    df = df.loc[(df['hit'] == 1) & (df['reject'] == 0) & (df['cond'] != 'X')]
    trials = {}
    for cond in conds:
        dfc = df.loc[df['cond']==cond]
        if len(dfc)>=thresh:
            indices = list(dfc['index'])
            eeg = epochs_np[indices]
            trials[cond] = eeg
    return trials

