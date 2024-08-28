### Code for P3: A Policy-Driven, Pace-Adaptive, and Diversity-Promoted Framework for Optimizing LLM Training
This is the repository for : https://arxiv.org/abs/2408.05541.
![image](framework.jpg)
This script supports the use of various selection strategies to optimize the LLM training process, including support for diversified selection mechanisms such as P3, SPL, Ramdom Selection.

## Prerequisites
Before running the script, ensure you have the following installed:
- Python>=3.8
- PyTorch>=2.0.1
- Transformers
- datasets
- NumPy
- tqdm

## How to run 
```
python train.py --sort [sort type] --select_num [selection number] --epoch [number of training epochs] --alpha [alpha value]
```
- sort: Specifies the method used for sorting or selecting data during training, with options including 'random', 'total', 'policy_sampling_SPL', and 'policy_sampling_SPL_DPP'.
- select_num: Sets the number of selections or samples used for training, default value is 10,000.
- epoch: The number of training epochs, default value is 5.
- alpha: Influences the control of the difficulty range.


## Additional Notes
Ensure all paths and configurations in the script are appropriately set according to your specific computing environment and dataset location.
The script may need parameter or configuration adjustments to optimize for different datasets.


## Citation
If you find this useful in your research, please consider citing
```
@misc{yang2024p3policydrivenpaceadaptivediversitypromoted,
      title={P3: A Policy-Driven, Pace-Adaptive, and Diversity-Promoted Framework for Optimizing LLM Training}, 
      author={Yingxuan Yang and Huayi Wang and Muning Wen and Weinan Zhang},
      year={2024},
      eprint={2408.05541},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.05541}, 
}
```
