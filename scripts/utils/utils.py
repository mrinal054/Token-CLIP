#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:30:08 2025

This code follows (D,H,W) format.

@author: M324371
"""

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import pyvista as pv


class Utils3D:
    @staticmethod
    def read_nifti(input_file):
        """Reads a NIfTI (.nii.gz) file and returns the image object, array, and metadata."""
        img_obj = sitk.ReadImage(input_file) # SITK shape: X, Y, Z
        img_arr = sitk.GetArrayFromImage(img_obj) # Numpy shape: Z, Y, X
        
        # Compute max and min intensity values
        max_intensity = np.max(img_arr)
        min_intensity = np.min(img_arr)
    
        metadata = {
            "Size": img_obj.GetSize(),  # Image dimensions in (X, Y, Z) order
            "Spacing": img_obj.GetSpacing(),  # Voxel size in each dimension (Sx, Sy, Sz)
            "Origin": img_obj.GetOrigin(),  # Physical world coordinate of the image origin (first voxel)
            "Direction": img_obj.GetDirection(),  # Orientation matrix (describes how the image is rotated in space)
            "PixelType": img_obj.GetPixelIDTypeAsString(),  # Data type of image pixels (e.g., int16, float32)
            "Dimensions": img_obj.GetDimension(),  # Number of spatial dimensions (typically 3 for 3D images)
            "MaxIntensity": max_intensity,  # Maximum pixel intensity in the image
            "MinIntensity": min_intensity,  # Minimum pixel intensity in the image
        }
        
        return img_obj, img_arr, metadata
    
    @staticmethod
    def write_nifti(output_file, img_arr, spacing, origin, direction):
        """
        Writes a NIfTI (.nii.gz) file from a NumPy array while preserving metadata.
    
        :param output_file: Path to save the .nii.gz file.
        :param img_arr: NumPy array (Z, Y, X) to be saved.
        :param spacing: Tuple containing voxel spacing (Sx, Sy, Sz).
        :param origin: Tuple containing the origin coordinates.
        :param direction: Tuple containing the direction matrix.
        """
        # Convert NumPy array back to SimpleITK image
        new_img = sitk.GetImageFromArray(img_arr)
    
        # Preserve metadata from the reference image
        new_img.SetSpacing(spacing)
        new_img.SetOrigin(origin)
        new_img.SetDirection(direction)
    
        # Save the image
        sitk.WriteImage(new_img, output_file)
        print(f"Saved NIfTI file to {output_file}")
    
    @staticmethod
    def resample(img_obj, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
        """
        Resamples an image to isotropic voxel spacing.

        :param img_obj: SimpleITK image object.
        :param new_spacing: Desired spacing in mm. Format: (Sx,Sy,Sz) or (Sw,Sh, Sd).
        :param interpolator: Interpolation method (default = sitk.sitkLinear, alternative: sitk.sitkNearestNeighbor).
        :return: Resampled SimpleITK image object and NumPy array.
        """

        assert len(new_spacing)==3, f"Length of new_spacing should be 3, but found {len(new_spacing)}."
        
        original_spacing = img_obj.GetSpacing()  # (Sx, Sy, Sz)
        original_size = img_obj.GetSize()  # (X, Y, Z)

        # Compute new size to maintain same physical dimensions
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)
        ]

        # Define resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img_obj.GetDirection())
        resampler.SetOutputOrigin(img_obj.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(interpolator)  # Linear interpolation by default

        # Apply resampling
        resampled_obj = resampler.Execute(img_obj)
        resampled_arr = sitk.GetArrayFromImage(resampled_obj)

        # print(f"Resampled to isotropic spacing: {new_spacing} mm")
        return resampled_obj, resampled_arr    

    @staticmethod
    def clip_intensity(volume, intensity_range:tuple):
        """
        Clip the volume intensities to the given range.

        :param volume: 3D numpy array.
        :param intensity_range: Tuple or list (min, max).
        :return: Clipped volume.
        """

        assert isinstance(intensity_range, (list, tuple)) and len(intensity_range) == 2, \
            "intensity_range must be a tuple or list of (min, max)."
        min_val, max_val = intensity_range
        return np.clip(volume, min_val, max_val)

    @staticmethod
    def normalize(volume):
        """
        Normalize the volume using its current min and max values.

        :param volume: 3D numpy array.
        :return: Normalized volume.
        """
        min_val = np.min(volume)
        max_val = np.max(volume)
        volume = (volume - min_val) / (max_val - min_val + 1e-8)
        return volume.astype(np.float32)      

    @staticmethod
    def resize(volume, desired_width: int = 128, desired_height: int = 128, desired_depth: int = 64, 
                       order: int = 1, original_spacing: tuple = None):
        """
        Resize the volume and optionally return the new spacing.
        
        Source: https://keras.io/examples/vision/3D_image_classification/
    
        :param volume: NumPy array representing the image volume.
        :param desired_width: Target width.
        :param desired_height: Target height.
        :param desired_depth: Target depth.
        :param order: Interpolation order (default=1 for trilinear interpolation).
        :param original_spacing: Optional original voxel spacing (tuple of 3 floats).
        
        :return: Resized image. If original_spacing is provided, returns (resized image, updated_spacing).
        
        Interpolation:
            order=0: Nearest-neighbor interpolation (fast but less accurate, can cause blocky artifacts).
            order=1: Trilinear interpolation (default, smoother than nearest-neighbor, balances speed and quality).
            order=2: Quadratic interpolation (higher accuracy but slightly slower).
            order=3: Cubic interpolation (even smoother but computationally more expensive).
        """
    
        # Get current depth, width, and height
        current_depth, current_height, current_width = volume.shape
    
        # Compute scaling factors
        depth_factor = current_depth / desired_depth
        width_factor = current_width / desired_width
        height_factor = current_height / desired_height
    
        # Resize the volume 
        vol_resized = ndimage.zoom(volume, (1 / depth_factor, 1 / height_factor, 1 / width_factor), order=order)
    
        # If original_spacing is provided, calculate new spacing
        if original_spacing:
            new_spacing = (
                original_spacing[0] * width_factor,
                original_spacing[1] * height_factor,
                original_spacing[2] * depth_factor
            )
            return vol_resized, new_spacing
    
        return vol_resized
    
    @staticmethod
    def resize_with_center_crop(
        volume: np.ndarray,
        desired_depth: int = 64,
        desired_height: int = 128,
        desired_width: int = 128,
        pad_value: int = 0
    ) -> np.ndarray:
        """
        Resize a volume by center-cropping or padding with a user-defined value to the desired shape.

        :param volume: (np.ndarray) Input 3D array of shape (D, H, W).
        :param desired_depth: (int) Target depth size (D).
        :param desired_height: (int) Target height size (H).
        :param desired_width: (int) Target width size (W).
        :param pad_value: (int) Value to use for padding. Default is 0.
        :return: (np.ndarray) Resized volume.
        """
        assert len(volume.shape) == 3, f"Expected 3D volume, got shape {volume.shape}"

        target_shape = (desired_depth, desired_height, desired_width)
        resized = np.full(target_shape, pad_value, dtype=volume.dtype)

        # Start with full volume and progressively crop each axis
        temp_volume = volume
        for axis, out_size in enumerate(target_shape):
            in_size = temp_volume.shape[axis]

            if in_size >= out_size:
                start_idx = (in_size - out_size) // 2
                end_idx = start_idx + out_size
                crop_slices = slice(start_idx, end_idx)
            else:
                crop_slices = slice(0, in_size)

            if axis == 0:
                temp_volume = temp_volume[crop_slices, :, :]
            elif axis == 1:
                temp_volume = temp_volume[:, crop_slices, :]
            else:
                temp_volume = temp_volume[:, :, crop_slices]

        # Compute starting indices for placing cropped volume
        start_indices = [(t - v) // 2 for t, v in zip(target_shape, temp_volume.shape)]
        end_indices = [start + v for start, v in zip(start_indices, temp_volume.shape)]

        resized[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] = temp_volume

        return resized

    @staticmethod
    def binary_mask(volume, keep_values:list):
        """
        Generate a binary mask from a 3D volume by retaining only specified label values.

        :param volume: (np.ndarray) 3D input array (D, H, W).
        :param keep_values: (list) List of integer values to retain.
        :return: (np.ndarray) Binary mask with 1s at specified values, 0 elsewhere.
        """
        volume = np.asarray(volume)
        mask = np.isin(volume, keep_values).astype(np.int8)

        return mask


    @staticmethod
    def labels_mapping(nii_path, save_path, selected_labels):
        """
        Process a .nii.gz file to keep only selected labels and renumber them sequentially. Useful for nnUNet.
    
        :param nii_path (str): Path to the input .nii.gz file.
        :param save_path (str): Path to save the processed .nii.gz file.
        :param selected_labels (list): List of label values to keep (e.g., [0, 2, 5]).
        """
        # Load the NIfTI file using SimpleITK
        sitk_img = sitk.ReadImage(nii_path)
        label_data = sitk.GetArrayFromImage(sitk_img).astype(np.int16)
    
        # Ensure selected_labels is sorted to maintain order
        selected_labels = sorted(set(selected_labels))  
    
        # Create a mapping of old labels to new sequential numbers
        new_labels = {old: new for new, old in enumerate(selected_labels)}
    
        # Process the label data
        new_label_data = np.zeros_like(label_data, dtype=np.int16)
        for old_label, new_label in new_labels.items():
            new_label_data[label_data == old_label] = new_label
    
        # Convert back to SimpleITK Image
        new_sitk_img = sitk.GetImageFromArray(new_label_data)
        new_sitk_img.CopyInformation(sitk_img)  # Preserve metadata
    
        # Save the new NIfTI file
        sitk.WriteImage(new_sitk_img, save_path)
    
        # Display new label mapping
        print(f"New label mapping: {new_labels}")

    # @staticmethod
    def visualizer(volume, spacing=(1, 1, 1), cmap='gray', opacity=[0.0, 0.0, 0.3, 0.7, 0.8, 0.3]):

        # Create grid from numpy directly using pv.wrap 
        grid = pv.wrap(volume)

        # Adjust spacing
        grid.spacing = spacing

        # Plot volume rendering
        plotter = pv.Plotter()
        plotter.add_volume(grid, cmap=cmap, opacity=opacity) 
        plotter.show()

        
#%% Test
if __name__ == "__main__":
    
    input_file="/research/m324371/Project/Digital_Twin/Segmentation/dataset/nnUNet_raw/Dataset009_DT_all_classes/imagesTr/MRN3885932_20090325_MOD-CT_ACC9737503-1_2_Abd-Pelvis-5-0-B40f_0000.nii.gz"
    label_file="/research/m324371/Project/Digital_Twin/Segmentation/dataset/nnUNet_raw/Dataset009_DT_all_classes/labelsTr/MRN3885932_20090325_MOD-CT_ACC9737503-1_2_Abd-Pelvis-5-0-B40f.nii.gz"
    
    utils3d = Utils3D()
    
    "Uncomment to read nifti file"
    img_obj, img_arr, metadata = utils3d.read_nifti(input_file)
    print(f"Shape of the original image in (D,H,W): {img_arr.shape}") # Shape (116,512,512)
    # print(metadata)
    # print(img_obj)

    "Uncomment to resample"
    # resampled_obj, resampled_arr = utils3d.resample(img_obj, new_spacing=(1,1,3), interpolator=sitk.sitkLinear)
    # print(f"Shape of the resampled image in (D,H,W): {resampled_arr.shape}") # Shape (193,380,380)

    "Uncomment to create a binary mask"
    # label_obj, label_arr, metadata = utils3d.read_nifti(label_file)
    # mask = utils3d.binary_mask(label_arr, keep_values=[1,2])
    # print(np.unique(resampled_arr))

    "Uncomment to remap labels"
    # selected_labels = [0, 1, 2, 5]  # Background (0), rk (1), lk (2), pancreas (5)
    # save_path = r"C:\MKD\MKmayo\MKprojects\MKdigitalTwin\Segmentation\relabeled.nii.gz"
    # utils3d.labels_mapping(label_file, save_path, selected_labels)
    
    "Uncomment to resize image"
    # resized_vol, updated_spacing = utils3d.resize(img_arr, 128, 128, 64, 1, metadata["Spacing"]) 
    # print("Shape of resized volume:", resized_vol.shape)

    "Uncomment to resize image with center crop"
    resized_vol = utils3d.resize_with_center_crop(img_arr, 64, 256, 256, -1000)

    
    "Uncomment to write in nifti"
    # utils3d.write_nifti("original_img.nii.gz", img_arr, metadata["Spacing"], metadata["Origin"], metadata["Direction"])
    # utils3d.write_nifti("isotropic_img.nii.gz", resampled_arr, resampled_obj.GetSpacing(), resampled_obj.GetOrigin(), resampled_obj.GetDirection())      
    # utils3d.write_nifti("resized_image.nii.gz", resized_vol, updated_spacing, metadata["Origin"], metadata["Direction"])
    # utils3d.write_nifti("mask.nii.gz", (mask*255).astype(np.uint8), metadata["Spacing"], metadata["Origin"], metadata["Direction"])

    "Uncomment to rander 3D data"
    Utils3D.visualizer(resized_vol, spacing=(1, 1, 1), cmap='gray', opacity=[0.0, 0.0, 0.3, 0.7, 0.8, 0.3])
    # Utils3D.visualizer(img_arr, cmap='gray', opacity=[0.0, 0.0, 0.3, 0.7, 0.8, 0.3])
    # Utils3D.visualizer(mask, cmap='binary', opacity=[0.0, 1.0])

# %%
