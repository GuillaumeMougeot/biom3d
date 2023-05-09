import numpy as np
import argparse

from biom3d.utils import abs_listdir, versus_one, dice

def eval(dir_lab, dir_out, num_classes):
    print("Start evaluation")
    paths_lab = [dir_lab, dir_out]
    list_abs = [sorted(abs_listdir(p)) for p in paths_lab]
    assert sum([len(t) for t in list_abs])%len(list_abs)==0, "[Error] Not the same number of labels and predictions! {}".format([len(t) for t in list_abs])

    results = []
    for idx in range(len(list_abs[0])):
        print("Metric computation for:", list_abs[1][idx])
        res = versus_one(
            fct=dice, 
            in_path=list_abs[1][idx], 
            tg_path=list_abs[0][idx], 
            num_classes=num_classes+1, 
            single_class=None)

        print("Metric result:", res)
        results += [res]
        
    print("Evaluation done! Average result:", np.mean(results))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Prediction evaluation.")
    parser.add_argument("-p", "--dir_pred", type=str, default=None,
        help="Path to the prediction directory")  
    parser.add_argument("-l", "--dir_lab", type=str, default=None,
        help="Path to the label directory")  
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    args = parser.parse_args()

    eval(args.dir_pred, args.dir_lab, args.num_classes)
