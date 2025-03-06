
import os
import boto3
from tqdm import tqdm


if __name__ == '__main__':
    save_path = r'C:\Users\agost\university\6\Project-Laboratory\project\fmri_data\data\fmri'

    files2download = [
        'MNINonLinear/Results/tfMRI_MOTOR_RL/tfMRI_MOTOR_RL.nii.gz',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/t.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/Sync.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/rh.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/rf.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/lh.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/lf.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/cue.txt',
        'MNINonLinear/Results/tfMRI_MOTOR_RL/MOTOR_run1_TAB.txt',
    ]
    
    
    s3_client = boto3.client('s3', aws_access_key_id= '', aws_secret_access_key='')
    bucket = 'hcp-openaccess'
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix ='HCP_1200/', Delimiter='/')
    
    failed_files = []
    subjects = []
    for prefix in objects['CommonPrefixes']:
        subjects.append(prefix['Prefix'])
    
    with tqdm(total=len(subjects), leave=True) as pbar:
        for subject_dir in subjects:
            pbar.set_description(f'Subject: {subject_dir.split("/")[1]}')
            subj_dir_save_pth = save_path + os.sep + subject_dir.split('/')[1]
            os.makedirs(subj_dir_save_pth, exist_ok=True)
            for file2download in files2download:
                file_save_path = subj_dir_save_pth + os.sep + file2download.split('/')[-1]
                if os.path.exists(file_save_path):
                    print(f'Existing data: {file_save_path}')
                    continue
                try:
                    with open(file_save_path, 'wb') as f:
                        s3_client.download_fileobj(bucket, subject_dir + file2download, f)
                except:
                    print(f'Could not download {subject_dir + file2download}!')
                    failed_files.append(subject_dir + file2download)
                    
            pbar.update(1)
            
    with open(r'C:\Users\agost\university\6\Project-Laboratory\project\fmri_data\data', 'w') as f:
        for failed_file in failed_files:
            f.write(failed_file)
            f.write('\n')
