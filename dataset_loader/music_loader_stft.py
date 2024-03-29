'''
Script with Pytorch's dataloader class
'''
import os
import glob
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import librosa.core as lc
import librosa
import csv
from utils.io import convert_audio_to_tensors


class MusicLoaderSTFT(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self, root_dir, snippet_size=512*128, split='train', transform=None, mel=False):
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split
    self.mel = mel

    self.sampling_rate = 8000  # Sorry, I'll fix this later
    self.n_fft = 512
    self.n_mels = 512

    self.mel_bank = librosa.filters.mel(
        self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels, norm=None)
    self.inv_mel_bank = np.linalg.pinv(self.mel_bank)

    self.snippet_size = snippet_size

    if split == 'train':
      self.curr_folder = self.train_folder
    elif split == 'test':
      self.curr_folder = self.test_folder

    # convert all mp3 files to torch tensors
    # TODO: optimize the conversion
    if len(glob.glob(os.path.join(root_dir, self.curr_folder, '*.pt'))) == 0:
      convert_audio_to_tensors(os.path.join(root_dir, self.curr_folder))

    self.file_names = glob.glob(
        os.path.join(root_dir, self.curr_folder, '*.pt')
    )

    self.data_len = len(self.file_names)

    self.labels = {}  # Create labels dict from CSV
    with open(os.path.join(root_dir, 'file_to_labels.csv'), 'r')as f:
      data = csv.reader(f)
      for row in data:
        self.labels[int(row[0])] = int(row[1])

  def spec_to_audio(self, spectrogram):
    if(self.mel):
      real = self.inv_mel_bank.dot(spectrogram[0, :, :])
      imag = self.inv_mel_bank.dot(spectrogram[1, :, :])
    else:
      real = spectrogram[0, :, :]
      imag = spectrogram[1, :, :]

    complex_spect = real + 1j * imag
    return lc.istft(complex_spect)

  def get_raw(self, index):
    # Get raw signal from data index

    raw_audio = torch.load(self.file_names[index])

    # randomly select a portion from the raw_audio
    start_idx = np.random.randint(0, raw_audio.numel()-self.snippet_size-1)
    snippet = raw_audio[start_idx:start_idx+self.snippet_size]

    return snippet.data.numpy()

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    signal = self.get_raw(index)
    ft = lc.stft(signal, n_fft=self.n_fft)

    spectrogram = np.zeros((2, ft.shape[0], ft.shape[1]), dtype="float32")
    spectrogram[0, :, :] = np.real(ft)
    spectrogram[1, :, :] = np.imag(ft)

    if(self.mel):
      real = self.mel_bank.dot(spectrogram[0, :, :])
      imag = self.mel_bank.dot(spectrogram[1, :, :])

      spectrogram = np.zeros(
          (2, real.shape[0], real.shape[1]), dtype="float32")
      spectrogram[0, :, :] = real
      spectrogram[1, :, :] = imag

    file_num = self.file_names[index].split('/')[-1]
    file_num = file_num.split('.')[0]
    label = self.labels[int(file_num)]

    return spectrogram, label

  def __len__(self):
    return self.data_len


# if __name__ == '__main__':

#   loader = MusicLoader('data/fma_xs', split='train')

#   for i in range(10):
#     a = loader[i]
#     signal = loader.get_raw(i)
#     reconstruct = loader.spec_to_audio(a, i)

#     # print(a.shape, "a shape")
#     # print(a[:100, :])
#     # a = np.repeat(a, 10, axis=1)

#     # plt.imshow(a)
#     # plt.show()

#     print(signal[:100])
#     print(reconstruct[:100])

#     plt.plot(signal)
#     plt.plot(reconstruct)
#     plt.show()
