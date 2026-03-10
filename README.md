# SceneGlue: Scene-Aware Transformer for Feature Matching without Scene-Level Annotation

<br/>


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
