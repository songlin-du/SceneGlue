# SceneGlue: Scene-Aware Transformer for Feature Matching without Scene-Level Annotation

<br/>

## Introduction
Local feature matching plays a critical role in understanding the correspondence between cross-view images. However, traditional methods are constrained by the inherent local nature of feature descriptors, limiting their ability to capture non-local scene information that is essential for accurate cross-view correspondence. In this paper, we introduce SceneGlue, a scene-aware feature matching framework designed to overcome these limitations. SceneGlue leverages a hybridizable matching paradigm that integrates implicit parallel attention and explicit cross-view visibility estimation. The parallel attention mechanism simultaneously exchanges information among local descriptors within and across images, enhancing the scene's global context. To further enrich the scene awareness, we propose the Visibility Transformer, which explicitly categorizes features into visible and invisible regions, providing an understanding of cross-view scene visibility. By combining explicit and implicit scene-level awareness, SceneGlue effectively compensates for the local descriptor constraints. Notably, SceneGlue is trained using only local feature matches, without requiring scene-level groundtruth annotations. This scene-aware approach not only improves accuracy and robustness but also enhances interpretability compared to traditional methods. Extensive experiments on applications such as homography estimation, pose estimation, image matching, and visual localization validate SceneGlue's superior performance.

```shell
# Paper
https://arxiv.org/abs/2604.13941
```


## Installation
```shell
conda env create -f environment.yaml
conda activate sceneglue
```

We provide the [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) to 
  - the scannet-1500-testset (~1GB).
  - the megadepth-1500-testset (~600MB).


## Test
You need to setup the testing subsets of ScanNet and MegaDepth first. We create symlinks from the previously downloaded datasets to `data/{{dataset}}/test`.

```shell
# set up symlinks
ln -s /path/to/scannet-1500-testset/* /path/to/LoFTR/data/scannet/test
ln -s /path/to/megadepth-1500-testset/* /path/to/LoFTR/data/megadepth/test
```

### MegaDepth
```shell
conda activate sceneglue
bash ./scripts/reproduce_test/outdoor.sh
```

### ScanNet
```shell
conda activate sceneglue
bash ./scripts/reproduce_test/indoor.sh
```
<br/>
