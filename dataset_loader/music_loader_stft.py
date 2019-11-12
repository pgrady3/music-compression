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
from utils.io import convert_audio_to_tensors


class MusicLoaderSTFT(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self, root_dir, split='train', snippet_size=16000, transform=None):
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

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

    # self.mean_val = torch.load(os.path.join(root_dir, 'mean.pt'))
    # self.std_val = torch.load(os.path.join(root_dir, 'std.pt'))

  def spec_to_audio(self, spectrogram, index=None):
    # If no index is given, then no phase information is reconstructed

    complex_spect = spectrogram[0, :, :] + 1j * spectrogram[1, :, :]
    return lc.istft(complex_spect)

  def get_raw(self, index):
    # Get raw signal from data index

    raw_audio = torch.load(self.file_names[index])

    # randomly select a portion from the raw_audio
    start_idx = np.random.randint(0, raw_audio.numel()-self.snippet_size-1)
    # start_idx = 50000  # Make deterministic so we can reconstruct

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
    ft = lc.stft(signal, n_fft=512)

    spectrogram = np.zeros((2, ft.shape[0], ft.shape[1]), dtype="float32")
    spectrogram[0, :, :] = np.real(ft)
    spectrogram[1, :, :] = np.imag(ft)

    return spectrogram

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
