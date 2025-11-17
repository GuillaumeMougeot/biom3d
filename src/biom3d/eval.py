"""
Evaluation module.

Used to compare predictions and groundtruth.

Examples
--------
.. code-block:: bash

    python -m biom3d.eval -p MyPred -l MyMasks --num_classes 2
    python -m biom3d.eval -p MyPred -l MyMasks -f IoU --num_classes 2

Or in python

.. code-block:: python

    print(eval("./MyPred","./MyMasks",2))
    print(eval("./MyPred","./MyMasks",2,iou))
"""

import numpy as np
import argparse

from biom3d.utils import versus_one, dice, iou,DataHandlerFactory

from typing import Callable

def robust_sort(str_list: list[str]) -> list[str]:
    """
    Perform a robust sorting of a list of strings, useful for sorting file paths.

    The sorting pads strings with zeros at the beginning so that all have the same length,
    then sorts lexicographically, and finally removes the padding.

    Parameters
    ----------
    str_list : list of str
        List of strings to sort.

    Returns
    -------
    list of str
        The sorted list of strings.
    """
    # max string lenght
    max_len = max([len(s) for s in str_list])

    # add zeros in the beginning so that all strings have the same length
    # associate with the original length to the elongated string
    same_len = {'0'*(max_len-len(s))+s:len(s) for s in str_list}

    # sort the dict by key
    sorted_same_len = {k:same_len[k] for k in sorted(same_len)}

    # remove zeros and return
    return [k[max_len-v:] for k,v in sorted_same_len.items()]

def eval(path_lab: str,
         path_out: str,
         num_classes: int,
         fct: Callable = dice,
         ) -> tuple[list[float], float]:
    """
    Evaluate segmentation results by comparing predictions to labels using a given metric.

    Parameters
    ----------
    path_lab : str
        Path to the folder containing label images.
    path_out : str
        Path to the folder containing predicted images.
    num_classes : int
        Number of classes for evaluation.
    fct : Callable, optional
        Metric function to compute (default is `dice`).

    Returns
    -------
    results: list of float
        List of metric results per image.
    mean: float
        Average of results
    """
    print("Start evaluation")
    handler1 = DataHandlerFactory.get(
        path_lab,
        read_only=True,
        eval='label',
    )

    handler2 = DataHandlerFactory.get(
        path_out,
        read_only=True,
        eval='pred',
    )
    assert len(handler1) == len(handler2), f"[Error] Not the same number of labels and predictions! '{len(handler1)}' for '{len(handler2)}'"

    results = []
    for (img1,_,_,),(img2,_,_) in zip(handler1,handler2):
        print("Metric computation for:", img1,img2)
        res = versus_one(
            fct=fct, 
            input_img=handler1.load(img1)[0],
            target_img=handler2.load(img2)[0],
            num_classes=num_classes+1, 
            single_class=None)

        print("Metric result:", res)
        results += [res]
        
    print("Evaluation done! Average {} result:".format(fct.__name__ ), np.mean(results))
    return results, np.mean(results)


if __name__=='__main__':
    supported_function = {
        "dice":dice,
        "equals":np.equal,
        "iou":iou,
    }
    parser = argparse.ArgumentParser(description="Prediction evaluation.")
    parser.add_argument("-p", "--path_pred","--dir_pred",dest="path_pred", type=str, default=None,
        help="Path to the prediction collection")  
    parser.add_argument("-l", "--path_lab","--dir_lab",dest="path_lab", type=str, default=None,
        help="Path to the label collection")  
    parser.add_argument("-f", "--function",dest="function", type=str, default='dice',
        help=f"(default=dice) Function used for evaluation. Supported : {', '.join(supported_function.keys())}")  
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    args = parser.parse_args()
    if args.function not in supported_function:
        print("Function '{}' not supported. Supported functions :'{}'".format(args.function,supported_function.keys()))
        exit(1)
    eval(args.path_lab, args.path_pred, args.num_classes,supported_function[args.function])
