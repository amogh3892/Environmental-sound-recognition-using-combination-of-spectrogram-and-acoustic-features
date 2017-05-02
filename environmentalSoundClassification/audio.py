import numpy as np
import librosa
from audioProcessingUtil import AudioProcessing
import SimpleITK as sitk
import sys


class Audio(object):

    def __init__(self,absFilePath,sampling_rate = None):

        self.folder = None
        self.fileName = None
        self.fileType = None
        self.denoised = None
        self.segments = None

        self.setFileProporties(absFilePath)
        self.data,self.fs = AudioProcessing.read(absFilePath,sr=None)
        self.duration = self.data.size/float(self.fs)

        self.segments = None

        # normalizing the sampling rate
        if sampling_rate is not None and self.fs != sampling_rate:
            self.data = AudioProcessing.resampleAudio(self.data,self.fs,sampling_rate)
            self.fs = sampling_rate

        # normalizing the amplitude (-1,1)
        self.data = AudioProcessing.rescaleAmplitude(self.data)



    def segment_audio(self,threshold = 5,max_segments = 2):
        segments =  AudioProcessing.segmentAudioByEnergyApproximation(self.data,self.fs,threshold=threshold,max_segments=max_segments)
        self.segments = segments
        return segments


    def setFileProporties(self,absFilePath):
        if "\\" in absFilePath:

            filetype = absFilePath.split("\\")[-1].split(".")[-1]
            filename = absFilePath.split("\\")[-1].split(".")[0]
            self.folder = absFilePath.rsplit('\\',1)[0]+"\\"
            self.fileType = filetype.lower()
            self.fileName = filename

        elif "/" in absFilePath:
            filetype = absFilePath.split("\\")[-1].split(".")[-1]
            filename = absFilePath.split("\\")[-1].split(".")[0]
            self.folder = absFilePath.rsplit('\\',1)[0]
            self.fileType = filetype.lower()
            self.fileName = filename

        else:
            print("Specify full path")
            sys.exit()


