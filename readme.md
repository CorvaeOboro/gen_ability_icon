# GEN_ABILITY_ICON
create ability icon images , a circular workflow of refinement using procgen augmented by neural networks .
- [DOWNLOAD ICONS](https://github.com/CorvaeOboro/gen_ability_icon/archive/refs/heads/master.zip) |  [VIEW ICONS](https://github.com/CorvaeOboro/gen_ability_icon/tree/master/icons#icons) |  [INSTALL](https://github.com/CorvaeOboro/gen_ability_icon#install)
- procedurally generated 3d renders using  [SideFX's Houdini](https://www.sidefx.com/) tools and [PDG TOPs](https://www.sidefx.com/products/pdg/) 
- mutated by text-to-image guided neural networks (  [VQGAN+CLIP](https://github.com/CompVis/taming-transformers) )
- cultivated dataset trained generative adversarial network to explore (  [STYLEGAN2ADA](https://github.com/NVlabs/stylegan2-ada) )

![00_icon_gen_20220407_comp](https://raw.githubusercontent.com/CorvaeOboro/gen_ability_icon/master/docs/00_icon_gen4_vqB_comp_0.jpg?raw=true "00_icon_gen_20220407_comp")

![00_icon_gen2_compA](https://raw.githubusercontent.com/CorvaeOboro/gen_ability_icon/master/docs/gen_ability_icon_process.jpg?raw=true "00_icon_gen2_compA")

# IMAGE DATASET
- a synthetic image dataset of circular magic ability icons
- collection of favored ability icons generated free to all 
- [DOWNLOAD ICONS](https://github.com/CorvaeOboro/gen_ability_icon/archive/refs/heads/master.zip) |  [VIEW ICONS](https://github.com/CorvaeOboro/gen_ability_icon/tree/master/icons#icons)

# STYLEGAN2ADA CHECKPOINT
- a stylegan2 network checkpoint trained on synthetic 256x256 images of generated selections .
- there is much to explore in 512-dimensional latent space , may you find favored .
- create new seeds using this notebook : [COLAB](https://colab.research.google.com/drive/1ZjLNer1pg6HluzxknAtPu0fQ2wDCvsxQ?authuser=1#scrollTo=t-prE54KutXG)
- [DOWNLOAD CHECKPOINT PKL](https://github.com/CorvaeOboro/gen_ability_icon/releases/download/gen_ability_icon_stylegan2ada_20220801/gen_ability_icon_stylegan2ada_20220801.pkl ) |  [GENERATE USING COLAB](https://colab.research.google.com/drive/1ZjLNer1pg6HluzxknAtPu0fQ2wDCvsxQ?authuser=1#scrollTo=t-prE54KutXG)

# PROCGEN
- houdini hda tool , z_ITEM_ABILITY_ICON.hda , generates 3d randomized icons of types ( slash , shatter , splatter ) 
- included houdini/GEN_ABILITY_ICON.hip file setup with [PDG TOPs](https://www.sidefx.com/products/pdg/) , renders randomized wedging  
- utilizes [SideFXLabs](https://github.com/sideeffects/SideFXLabs) hda tools and [ZENV](https://github.com/CorvaeOboro/zenv) hda tools 
- focused on volumetric lighting , metallic material , randomized vertex color

![gen_ability_icon_pdg_02](https://raw.githubusercontent.com/CorvaeOboro/gen_ability_icon/master/docs/gen_ability_icon_pdg_02.jpg?raw=true "gen_ability_icon_pdg_02")

![00_icon_procgen_comp](https://raw.githubusercontent.com/CorvaeOboro/gen_ability_icon/master/docs/00_icon_procgen_comp.jpg?raw=true "00_icon_procgen_comp")

# GUIDED MUTATION / REMIXING
- with initial set of procgen selected , expand the dataset and alter using various techniques :
-  [VQGAN+CLIP](https://github.com/CompVis/taming-transformers)  - text-to-image guided modification of input image , similar to text based styletransfer . 
- text prompts used for text guided image mutation included in [prompts_list.txt](https://github.com/CorvaeOboro/gen_ability_icon/blob/master/prompts_list.txt)
- IMAGE_COLLAGE.py - given a folder of images randomly composites them with randomized hue / brightness / normalization 

![00_icon_gen2_compB](https://raw.githubusercontent.com/CorvaeOboro/gen_ability_icon/master/docs/00_icon_gen2_compB.jpg?raw=true "00_icon_gen2_compB")

# INSTALL

```.bash
# create anaconda env from included environment.yml
git clone 'https://github.com/CorvaeOboro/gen_ability_icon'
cd gen_ability_icon
conda env create --prefix venv -f environment.yml
conda activate venv

# clone STYLEGAN2ADA
git clone 'https://github.com/NVlabs/stylegan2-ada'

# clone VQGAN+CLIP 
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'

# download VQGAN checkpoint imagenet 16k
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384

```

# WORKFLOW
- generate procgen renders from houdini , selecting favored renders
- mutate those renders via text guided VQGAN+CLIP 
- combine the renders and mutants via random collaging 
- select the favored icons to create a stylegan2 dataset 
- train stylegan2 network , then generate seeds from trained checkpoint
- cultivate the complete dataset thru selection and art direction adjustments 
- repeat to expand and refine by additional text guided mutation , retraining , regenerating

# THANKS
many thanks to 
- NVIDIA NVLabs - https://github.com/NVlabs/stylegan2-ada
- CLIP - https://github.com/openai/CLIP
- VQGAN - https://github.com/CompVis/taming-transformers
- Katherine Crowson : [https://github.com/crowsonkb](https://github.com/crowsonkb)  https://arxiv.org/abs/2204.08583
- NerdyRodent : https://github.com/nerdyrodent/VQGAN-CLIP

# AKNOWLEDGEMENTS
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@misc{https://doi.org/10.48550/arxiv.2103.00020,
  doi = {10.48550/ARXIV.2103.00020},
  url = {https://arxiv.org/abs/2103.00020},
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# ATTRIBUTION
- the following icons were initially apart of the training dataset tho not included in the current set :
- J. W. Bjerk (eleazzaar) -- www.jwbjerk.com/art  ( CreativeCommons 3.0 )
- Lorc ( CreativeCommons 3.0 )

# CREATIVE COMMONS ZERO 
free to all , [creative commons CC0](https://creativecommons.org/publicdomain/zero/1.0/) , free to redistribute , no attribution required