# import sys
# sys.path.append("/research/m324371/LibraryMKD/utils/")

from utils import Utils3D  # before token selection, it was - from utils import Utils3D  
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import pandas as pd

class CTReportDatasetKLab(Dataset):
    def __init__(self,
                 report_file,
                 text_column,
                 img_dir,
                 resample=(1.0, 1.0, 1.0),
                 n_zSlices=None,
                 zSlices_pad_value=0,
                 clip=(-1000, 400),
                 clip_percentile:tuple=None,
                 normalize=True,
                 resize_shape=(64, 128, 128),
                 transform=None,
                 verbose=False):
        """
        Dataset returning image and radiology report text.

        :param report_file: DataFrame.
        :param text_column: Name of the column that contains text
        :param img_dir: Path to image folder.
        :param resample: Resample spacing (D, H, W)
        :param n_zSlices: Target number of z-slices
        :param zSlices_pad_value: Value for padding
        :param clip: Clip HU values
        :param clip_percentile: Tuple to clip image intensities based on percentile value. Only works when clip is set to None. 
        :param normalize: Normalize HU values
        :param resize_shape: (D, H, W)
        :param transform: Optional transforms
        :param verbose: Print debug info
        """
        self.df = pd.read_excel(report_file).reset_index(drop=True)
        self.text_column = text_column
        self.img_dir = img_dir
        self.resample = resample
        self.n_zSlices = n_zSlices
        self.zSlices_pad_value = zSlices_pad_value
        self.clip = clip
        self.clip_percentile = clip_percentile
        self.normalize = normalize
        self.resize_shape = resize_shape
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Filename']
        img_path = os.path.join(self.img_dir, img_name)

        # Read image
        img_obj, img_arr, metadata = Utils3D.read_nifti(img_path)

        if img_arr.ndim == 4:
            img_arr = img_arr[0]
            size = list(img_obj.GetSize())
            size[-1] = 0
            index = [0] * len(size)
            img_obj = sitk.Extract(img_obj, size=size, index=index)

        # Resample
        if self.resample:
            spacing = list(reversed(self.resample))
            for i in range(3):
                if spacing[i] is None:
                    spacing[i] = img_obj.GetSpacing()[i]
            img_obj, img_arr = Utils3D.resample(img_obj, spacing)

        # Pad
        if self.n_zSlices:
            current_depth = img_arr.shape[0]
            if current_depth < self.n_zSlices:
                pad_total = self.n_zSlices - current_depth
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                img_arr = np.pad(img_arr, ((pad_before, pad_after), (0, 0), (0, 0)),
                                 mode='constant', constant_values=self.zSlices_pad_value)

        if self.clip:
            img_arr = Utils3D.clip_intensity(img_arr, self.clip)
        elif self.clip_percentile:
            lower_p = np.percentile(img_arr, self.clip_percentile[0])
            upper_p = np.percentile(img_arr, self.clip_percentile[1])
            img_arr = np.clip(img_arr, lower_p, upper_p)

            if self.verbose: print("Percentile values:", lower_p, upper_p)
                
        # if self.normalize:
        #     if self.clip is not None:
        #         max_clip = self.clip[-1]
        #         assert max_clip != 0, "Clip max value is 0. Division by zero is not allowed."
        #         img_arr = (img_arr / max_clip).astype(np.float32)  # <<<<<<<<<<<<<<<<<<<<<<<<< replicating the behavior of CT-CLIP
        #     else:
        #         img_arr = Utils3D.normalize(img_arr)

        if self.normalize:
            img_arr = Utils3D.normalize(img_arr)

        if self.resize_shape:
            img_arr, _ = Utils3D.resize(
                img_arr,
                desired_width=self.resize_shape[2],
                desired_height=self.resize_shape[1],
                desired_depth=self.resize_shape[0],
                order=1,
                original_spacing=metadata["Spacing"]
            )

        if self.transform:
            img_arr = self.transform(img_arr)

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()  # Shape: [1, D, H, W]
        report_text = str(row[self.text_column])

        # Do some cleanup like in CT-CLIP
        report_text = report_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')

        return img_tensor, report_text


# Uncomment to check the dataset
# if __name__ == "__main__":

#     report_file = "/research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_791v2-train.xlsx"
    
#     img_dir = "/research/m324371/Project/Digital_Twin/Classification/Dataset_791/"

#     dataset = CTReportDatasetKLab(report_file=report_file, 
#                                 img_dir=img_dir,
#                                 text_column='Radiology Report',
#                                 resample=(1.0, None, None),
#                                 n_zSlices=None,
#                                 zSlices_pad_value=0,
#                                 clip=(-1000, 400),
#                                 normalize=True,
#                                 resize_shape=(64, 128, 128),
#                                 transform=None,
#                                 verbose=False
#                             )
#     img, text = dataset[0]
#     print(img.shape)  # torch.Size([1, D, H, W])
#     print(text)       # The radiology report



class CTReportDatasetKLabCTMR(Dataset):
    def __init__(self,
                 report_file,
                 text_column,
                 resample=(1.0, 1.0, 1.0),
                 n_zSlices=None,
                 zSlices_pad_value=0,
                 clip=(-1000, 400),
                 clip_percentile:tuple=None,
                 normalize=True,
                 resize_shape=(64, 128, 128),
                 transform=None,
                 verbose=False):
        """
        Dataset returning image and radiology report text.

        :param report_file: DataFrame.
        :param text_column: Name of the column that contains text
        :param resample: Resample spacing (D, H, W)
        :param n_zSlices: Target number of z-slices
        :param zSlices_pad_value: Value for padding
        :param clip: Clip HU values
        :param clip_percentile: Tuple to clip image intensities based on percentile value. Only works when clip is set to None. 
        :param normalize: Normalize HU values
        :param resize_shape: (D, H, W)
        :param transform: Optional transforms
        :param verbose: Print debug info
        """
        self.df = pd.read_excel(report_file).reset_index(drop=True)
        self.text_column = text_column
        self.resample = resample
        self.n_zSlices = n_zSlices
        self.zSlices_pad_value = zSlices_pad_value
        self.clip = clip # will be used for CT
        self.clip_percentile = clip_percentile # will be used for MR
        self.normalize = normalize
        self.resize_shape = resize_shape
        self.transform = transform
        self.verbose = verbose

        # Make sure necessary columns exist
        assert "Modality" in self.df.columns, "DataFrame is missing the Modality column."
        assert "Filename" in self.df.columns, "DataFrame is missing the Filename column."
        assert "Directories" in self.df.columns, "DataFrame is missing the Directories column."
        assert self.text_column in self.df.columns, f"DataFrame is missing the {self.text_column} column."

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        modality = row['Modality']
        img_path = row['Directories'] # full image path

        # Read image
        img_obj, img_arr, metadata = Utils3D.read_nifti(img_path)

        if img_arr.ndim == 4:
            img_arr = img_arr[0]
            size = list(img_obj.GetSize())
            size[-1] = 0
            index = [0] * len(size)
            img_obj = sitk.Extract(img_obj, size=size, index=index)

        # Resample
        if self.resample:
            spacing = list(reversed(self.resample))
            for i in range(3):
                if spacing[i] is None:
                    spacing[i] = img_obj.GetSpacing()[i]
            img_obj, img_arr = Utils3D.resample(img_obj, spacing)

        # Pad
        if self.n_zSlices:
            current_depth = img_arr.shape[0]
            if current_depth < self.n_zSlices:
                pad_total = self.n_zSlices - current_depth
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                img_arr = np.pad(img_arr, ((pad_before, pad_after), (0, 0), (0, 0)),
                                 mode='constant', constant_values=self.zSlices_pad_value)

        # Clip intensity
        if modality == "CT": 
            img_arr = Utils3D.clip_intensity(img_arr, self.clip)
        elif modality == "MR":
            lower_p = np.percentile(img_arr, self.clip_percentile[0])
            upper_p = np.percentile(img_arr, self.clip_percentile[1])
            img_arr = np.clip(img_arr, lower_p, upper_p)

            if self.verbose: print("Percentile values:", lower_p, upper_p)
        else:
            raise ValueError(f"Unknown modality ({modality}). It should be either CT or MR.")
                
        # if self.normalize:
        #     if self.clip is not None:
        #         max_clip = self.clip[-1]
        #         assert max_clip != 0, "Clip max value is 0. Division by zero is not allowed."
        #         img_arr = (img_arr / max_clip).astype(np.float32)  # <<<<<<<<<<<<<<<<<<<<<<<<< replicating the behavior of CT-CLIP
        #     else:
        #         img_arr = Utils3D.normalize(img_arr)

        if self.normalize:
            img_arr = Utils3D.normalize(img_arr)

        if self.resize_shape:
            img_arr, _ = Utils3D.resize(
                img_arr,
                desired_width=self.resize_shape[2],
                desired_height=self.resize_shape[1],
                desired_depth=self.resize_shape[0],
                order=1,
                original_spacing=metadata["Spacing"]
            )

        if self.transform:
            img_arr = self.transform(img_arr)

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()  # Shape: [1, D, H, W]
        report_text = str(row[self.text_column])

        # Do some cleanup like in CT-CLIP
        report_text = report_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')

        return img_tensor, report_text