import pandas as pd
import librosa
import os

def get_files(path):
    '''
    Get all files from a directory
    '''
    files = []
    for root, dirs, file in os.walk(path):
        for f in file:
            files.append(os.path.join(root, f))
    return files


class Dataset:
    def __init__(self,path='data'):
        pos = {
            0:'Modality',
            1:'Type',
            2:'Emotion',
            3:'Intensity',
            4:'Statement',
            5:'Repetition',
            6:'Actor',
        }
        self.df = pd.DataFrame({'Path':get_files(path)})
        for i in pos:
            self.df[pos[i]] = self.df['Path'].apply(lambda x: int(x.split('/')[-1].split('.')[0].split('-')[i])-1)
        
        self.df['Audio'] = self.df['Path'].apply(lambda x: librosa.load(x, sr=16000)[0])
    
    def __len__(self):
        return len(self.df)

    def head(self):
        return self.df.head()
    
    def __call__(self):
        return self.df


