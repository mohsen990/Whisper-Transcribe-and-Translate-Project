#  python -m  pip install git+https://github.com/openai/whisper.git
#  Notice you should use same python interpreter for installing and using in IED For Change use ( Ctrl + Shift + P ) choose proper one.
#  It should be same in cmd command and IDE for example both python 3.12.1 check versions 
   
# python -m pip install ffmpeg
# Download ffmpeg from link https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z
# Extract zip file and rename file to ffmpeg and copy to root C:/
# run the command    setx /m PATH "C:\ffmpeg\bin;%PATH%"   in the CMD
# restart the computer 

import whisper
from whisper.utils import get_writer
import os
import pandas as pd
from pytube import YouTube

# Function to find directory 
def find_path(file_name):
   current_dir = ''
   current_dir = os.path.dirname(os.path.abspath(__file__))
   current_dir = current_dir +  "Data\\"
   file_path = os.path.join(current_dir, file_name)
   return file_path

# Function to transcribe Audio file 
def transcribe_audio(FilePath):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio=FilePath , fp16=False)
    return result

def translate_audio(FilePath):
    model = whisper.load_model("tiny")
    results = model.transcribe(audio=FilePath, language="En", task="translate",fp16=False)
    return results

# Function to write transcribe in files with special format 
def save_file(results, Pre , format='tsv'):
    writer = get_writer(format, './output/')
    writer(results, f'transcribe_audio'+ Pre +'.{format}')



filename = "english_padcast.mp3"
filepath = './Data/' + filename

output = transcribe_audio(filepath)
#save_file(output)
save_file(output,'_En', 'txt')
save_file(output,'_En', 'srt')



germanFile = "erdbeben-auf-haiti.mp3"
filepath2 = './Data/' + germanFile

output_JP = translate_audio(filepath2)
#save_file(output)
save_file(output_JP,'_Gr', 'txt')
save_file(output_JP,'_Gr', 'srt')

videoFile = "sampleVideo.mp4"
filepath3 = './Data/' + videoFile
resuldVideo = transcribe_audio(filepath3)
#save_file(output)
save_file(resuldVideo,'_Vd', 'txt')
save_file(resuldVideo,'_Vd', 'srt')


print(output["text"])
speech = pd.DataFrame.from_dict(output['segments'])
speechText = speech[['id', 'seek','start','end','text' ]]
print(speechText.head(10))




