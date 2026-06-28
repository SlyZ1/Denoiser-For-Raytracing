from torch.utils.data import Dataset
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

def readExr(path):
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        print(path)
        raise FileNotFoundError()

class BMFRDataset(Dataset):
    def __init__(self, path="./data", num_frames=60, crop_size=128):
        super().__init__()
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.scene_paths = []
        for folder in os.listdir(path):
            self.scene_paths.append(os.path.join(path, folder, "inputs/"))

    def __len__(self):
        return len(self.scene_paths) * self.num_frames

    def __getitem__(self, index):
        scene = index // self.num_frames
        frame = index - scene * self.num_frames
        scene_path = self.scene_paths[scene]
        crop = np.array([np.random.randint(0, 720 - self.crop_size), np.random.randint(0, 1280 - self.crop_size)])
        albedo = readExr(os.path.join(scene_path, f"albedo{frame}.exr"))[crop[0]:crop[0]+self.crop_size, crop[1]:crop[1]+self.crop_size]
        color = readExr(os.path.join(scene_path, f"color{frame}.exr"))[crop[0]:crop[0]+self.crop_size, crop[1]:crop[1]+self.crop_size]
        reference = readExr(os.path.join(scene_path, f"reference{frame}.exr"))[crop[0]:crop[0]+self.crop_size, crop[1]:crop[1]+self.crop_size]
        shading_normal = readExr(os.path.join(scene_path, f"shading_normal{frame}.exr"))[crop[0]:crop[0]+self.crop_size, crop[1]:crop[1]+self.crop_size]
        #world_position = readExr(os.path.join(scene_path, f"world_position{frame}.exr"))[crop[0]:crop[0]+self.crop_size, crop[1]:crop[1]+self.crop_size]
        
        input = np.concat([albedo, np.log1p(color), shading_normal], axis=-1)
        input = np.permute_dims(input, (2, 0, 1))
        reference = np.permute_dims(reference, (2, 0, 1))
        return input, reference

        