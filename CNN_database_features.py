from PIL import Image
from pathlib import Path
import numpy as np

from CNN_feature_extractor import FeatureExtractor

# extract the features of the images in database by CNN
if __name__ == "__main__":
    fe = FeatureExtractor()
    for img_path in sorted(Path("static/img").glob("*.jpg")):
        print(img_path)

        # Extract deep feature hear
        feature = fe.extract(img=Image.open(img_path))
        print(type(feature), feature.shape)

        feature_path = Path("static/feature") / (img_path.stem + ".npy")
        print(feature_path)

        np.save(f"{feature_path}", feature)

