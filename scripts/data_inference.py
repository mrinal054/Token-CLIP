# import sys
# sys.path.append("/research/m324371/LibraryMKD/utils/")

import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils import Utils3D # before token selection, it was - from utils import Utils3D


class CTReportDatasetinferKLab(Dataset):
    def __init__(self,
                 img_dir,
                 report_file,
                 text_column,
                 label_file,
                 label_cols: list,
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

        :param img_dir: Folder containing NIfTI images.
        :param report_file: Excel file with 'Names' and 'text' columns (can have other metadata).
        :param text_column: Name of the column that contains text
        :param label_file: Excel file with 'Names' and one-hot label columns.
        :param label_cols: List of column names in the DataFrame to use as labels. 
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
        self.text_column=text_column
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

        # Read both Excel files
        self.report_df = pd.read_excel(report_file).reset_index(drop=True)
        self.label_df = pd.read_excel(label_file).reset_index(drop=True)

        # Check length match
        assert len(self.report_df) == len(self.label_df), \
            f"Length mismatch: report_df={len(self.report_df)}, label_df={len(self.label_df)}"

        # Check name alignment
        for i in range(len(self.report_df)):
            name1 = self.report_df.iloc[i]['Filename']
            name2 = self.label_df.iloc[i]['Filename']
            assert name1 == name2, f"Mismatch at index {i}: {name1} != {name2}"

        # Extract values from label columns
        # label_cols = self.label_df.columns[1:]
        self.label_df['one_hot_labels'] = list(self.label_df[label_cols].values)

    def __len__(self):
        return len(self.report_df)

    def __getitem__(self, idx):
        img_name = self.report_df.iloc[idx]['Filename']
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

        # Pad slices
        if self.n_zSlices:
            current_depth = img_arr.shape[0]
            if current_depth < self.n_zSlices:
                pad_total = self.n_zSlices - current_depth
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                img_arr = np.pad(img_arr, ((pad_before, pad_after), (0, 0), (0, 0)),
                                 mode='constant', constant_values=self.zSlices_pad_value)

        if self.clip is not None:
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

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()  # [1, D, H, W]

        # Extract report text and label
        report_text = str(self.report_df.iloc[idx][self.text_column])

        # Do some cleanup like in CT-CLIP
        report_text = report_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')

        # Collect onehot labels
        onehotlabels = self.label_df.iloc[idx]['one_hot_labels'] 
        # print("---", onehotlabels) # [1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]

        # Use name as the accession. Checked CT-CLIP, not something significant.
        name_acc = str(self.report_df.iloc[idx]['Filename'])
        # print("---", name_acc) # MRN6420337_20190910_MOD-MR_ACC12286742_3_Cor-SSFSE-FS-Kid-Vol.nii.gz

        return img_tensor, report_text, onehotlabels, name_acc
    
# #%% Example case
# if __name__ == '__main__':

#     from torch.utils.data import DataLoader

#     img_dir = "/research/m324371/Project/Digital_Twin/Classification/Dataset_791/"
#     report_file = "/research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_791v2-val.xlsx"   # must contain 'Names' and 'text'
#     label_file = "/research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_791v2_labels-val.xlsx"         # must contain 'Names' + label columns

#     # Create dataset
#     dataset = CTReportDatasetinferKLab(
#         img_dir=img_dir,
#         report_file=report_file,
#         text_column='Radiology Report',
#         label_file=label_file,
#         resample=(1.0, 1.0, 1.0),
#         n_zSlices=None,
#         zSlices_pad_value=0,
#         clip=(-1000, 400),
#         normalize=True,
#         resize_shape=(64, 128, 128),
#         transform=None,
#         verbose=False
#     )

#     # Wrap with DataLoader
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     itr = iter(dataloader)

#     #%%
#     image_tensor, report_text, one_hot_labels, name_acc = next(itr)

#     print("Image Tensor Shape:", image_tensor.shape)         # torch.Size([1, 1, 64, 128, 128])
#     print("Report Text:", report_text)
#     print("One-Hot Labels:", one_hot_labels) # tensor([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
#     print("Accession Name:", name_acc) # ('MRN6234327_201324910_MOD-MR_ACC123423442_3_Cor-SSFSE-FS-Kid-Vol.nii.gz',)




class CTReportDatasetinferKLabCTMR(Dataset):
    def __init__(self,             
                 report_file,
                 text_column,
                 label_file,
                 label_cols: list,
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

        :param report_file: Excel file with 'Names' and 'text' columns (can have other metadata).
        :param text_column: Name of the column that contains text
        :param label_file: Excel file with 'Names' and one-hot label columns.
        :param label_cols: List of column names in the DataFrame to use as labels. 
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
        self.text_column=text_column
        self.resample = resample
        self.n_zSlices = n_zSlices
        self.zSlices_pad_value = zSlices_pad_value
        self.clip = clip
        self.clip_percentile = clip_percentile
        self.normalize = normalize
        self.resize_shape = resize_shape
        self.transform = transform
        self.verbose = verbose

        # Read both Excel files
        self.report_df = pd.read_excel(report_file).reset_index(drop=True)
        self.label_df = pd.read_excel(label_file).reset_index(drop=True)

        # Make sure necessary columns exist
        assert "Modality" in self.report_df.columns, "DataFrame is missing the Modality column."
        assert "Filename" in self.report_df.columns, "DataFrame is missing the Filename column."
        assert "Directories" in self.report_df.columns, "DataFrame is missing the Directories column."
        assert self.text_column in self.report_df.columns, f"DataFrame is missing the {self.text_column} column."

        # Check length match
        assert len(self.report_df) == len(self.label_df), \
            f"Length mismatch: report_df={len(self.report_df)}, label_df={len(self.label_df)}"

        # Check name alignment
        for i in range(len(self.report_df)):
            name1 = self.report_df.iloc[i]['Filename']
            name2 = self.label_df.iloc[i]['Filename']
            assert name1 == name2, f"Mismatch at index {i}: {name1} != {name2}"

        # Extract values from label columns
        # label_cols = self.label_df.columns[1:]
        self.label_df['one_hot_labels'] = list(self.label_df[label_cols].values)

    def __len__(self):
        return len(self.report_df)

    def __getitem__(self, idx):
        report_row = self.report_df.iloc[idx]
        modality = report_row["Modality"]
        img_path = report_row['Directories'] # full image path

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

        # Pad slices
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

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()  # [1, D, H, W]

        # Extract report text and label
        report_text = str(report_row[self.text_column])

        # Do some cleanup like in CT-CLIP
        report_text = report_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')

        # Collect onehot labels
        onehotlabels = self.label_df.iloc[idx]['one_hot_labels'] 
        # print("---", onehotlabels) # [1 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]

        # Use name as the accession. Checked CT-CLIP, not something significant.
        name_acc = str(report_row['Filename'])
        # print("---", name_acc) # MRN6420337_20190910_MOD-MR_ACC12286742_3_Cor-SSFSE-FS-Kid-Vol.nii.gz

        return img_tensor, report_text, onehotlabels, name_acc