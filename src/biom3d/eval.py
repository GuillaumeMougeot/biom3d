import numpy as np
import argparse

from biom3d.utils import versus_one, dice, iou,DataHandlerFactory

def robust_sort(str_list):
    """Robust sorting of string. Useful for list of paths sorting.
    """
    # max string lenght
    max_len = max(list(len(s) for s in str_list))

    # add zeros in the beginning so that all strings have the same length
    # associate with the original length to the elongated string
    same_len = {'0'*(max_len-len(s))+s:len(s) for s in str_list}

    # sort the dict by key
    sorted_same_len = {k:same_len[k] for k in sorted(same_len)}

    # remove zeros and return
    return [k[max_len-v:] for k,v in sorted_same_len.items()]

def eval(path_lab, path_out, num_classes,fct=dice):
    print("Start evaluation")
    handler1 = DataHandlerFactory.get(
        path_lab,
        read_only=True,
        img_path = path_lab,
    )
    handler2 = DataHandlerFactory.get(
        path_out,
        read_only=True,
        img_path = path_out,
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
        
    print("Evaluation done! Average result:", np.mean(results))
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
    parser.add_argument("-f", "--function",dest="function", type=str, default=dice,
        help=f"(default=dice) Function used for evaluation. Supported : {', '.join(supported_function.keys())}")  
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    args = parser.parse_args()
    if args.function not in supported_function:
        print("Function '{}' not supported. Supported functions :'{}'".format(args.function,supported_function.keys()))
        exit(1)
    eval(args.path_pred, args.path_lab, args.num_classes,supported_function[args.function])
