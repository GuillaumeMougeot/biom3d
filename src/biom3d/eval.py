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
        results += [versus_one(
            fct=dice, 
            in_path=list_abs[1][idx], 
            tg_path=list_abs[0][idx], 
            num_classes=num_classes, 
            single_class=None)]
        print("Metric result:", print(results[-1]))
    print("Evaluation done! Average result:", np.mean(results))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Dataset preprocessing for training purpose.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--max_dim", type=int, default=128,
        help="Maximum size of one dimension of the patch (default: 128)")  
    # parser.add_argument("--min_dis", default=False,  action='store_true', dest='min_dis',
    #     help="Minimal display. Display only the raw batch, aug_patch, patch and pool")
    parser.add_argument("--spacing", default=False,  action='store_true', dest='spacing',
        help="Print median spacing if set.")
    parser.add_argument("--median", default=False,  action='store_true', dest='median',
        help="Print the median.")
    args = parser.parse_args()


    # if args.min_dis:
    #     minimal_display(img_dir=args.img_dir, max_dims=(args.max_dim, args.max_dim, args.max_dim))
    # else: 
    median = compute_median(path=args.img_dir, return_spacing=args.spacing)
    
    if args.spacing: 
        median_spacing = median[1]
        median = median[0]
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))

    display_info(patch, pool, batch)
    
    if args.spacing:print("MEDIAN_SPACING =",list(median_spacing))
    if args.median:print("MEDIAN =", list(median))
