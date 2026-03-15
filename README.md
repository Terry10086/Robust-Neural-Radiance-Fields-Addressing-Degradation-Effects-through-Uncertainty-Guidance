## Description / Implementation

Code for **Uncertainty-guided Neural Radiance Fields against Diverse Degradation Factors**.

The implementation is built upon:
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), corresponding to the **Ours (NeRF)** model in the paper.
- [NeuS](https://github.com/Totoro97/NeuS), corresponding to the **Ours (NeuS)** model in the paper.

The installation procedure follows the instructions provided in the original NeRF and NeuS repositories.


## How To Run?

```
# Ours (NeRF)
python run_nerf.py 
# Ours (NeuS)
python exp_runner.py
```


## Citation
Kudos to the authors for their amazing results:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
```
@article{wang2021neus,
  title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2106.10689},
  year={2021}
}
```
