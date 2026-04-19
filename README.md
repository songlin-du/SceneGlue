# SceneGlue: Scene-Aware Transformer for Feature Matching without Scene-Level Annotation

**IEEE Transactions on Circuits and Systems for Video Technology**, 2026
<br/>
## Introduction
SceneGlue is a scene-aware feature matching framework that overcomes the locality limitation of traditional descriptors by integrating parallel attention for implicit global context modeling and a Visibility Transformer for explicit cross-view visibility estimation. By jointly leveraging implicit and explicit scene-level awareness without requiring scene-level annotations, it significantly improves matching accuracy, robustness, and interpretability across multiple vision tasks.
```shell
https://ieeexplore.ieee.org/document/11483154
https://arxiv.org/abs/2604.13941
```
---

## What makes SceneGlue special?

- SceneGlue goes beyond traditional point-level matching by introducing **scene-aware feature matching**, integrating global scene information with local correspondences.  
- It combines **implicit scene modeling (via parallel self- and cross-attention)** with **explicit reasoning (via a Visibility Transformer for cross-view visibility estimation)**.  
- The proposed **parallel attention mechanism** enables more efficient and richer interactions than the sequential attention used in prior methods.  
- SceneGlue also improves feature representation through **multi-scale descriptors and wave-based position encoding**, enhancing robustness to scale and geometry variations.  
- Importantly, it learns scene-level awareness **without requiring scene-level annotations**, achieving better accuracy, robustness, and interpretability than existing methods.  

---

## Installation
```shell
conda env create -f environment.yaml
conda activate sceneglue
```

We provide the [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) to 
  - the scannet-1500-testset (~1GB).
  - the megadepth-1500-testset (~600MB).

---

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

---

## Citation

```bibtex
@ARTICLE{SceneGlue,
  author={Du, Songlin and Lu, Xiaoyong and Yan, Yaping and Xiao, Guobao and Lu, Xiaobo and Ikenaga, Takeshi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SceneGlue: Scene-Aware Transformer for Feature Matching without Scene-Level Annotation}, 
  year={2026},
  doi={10.1109/TCSVT.2026.3684799}
}
```

<br/>
