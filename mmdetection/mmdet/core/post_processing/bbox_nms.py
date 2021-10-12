import torch

from mmdet.ops.nms import batched_nms


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   stg='std'):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    #the predicted box before score filter.
    bboxes_pre = bboxes.reshape(-1,4) # n,cls,4
    scores_pre = scores.reshape(-1)
    bboxes_pre = torch.cat([bboxes_pre, scores_pre[:, None]], -1)
    labels_pre = scores.nonzero(as_tuple=False)[:,1]

    bboxes = bboxes[valid_mask] # remove  low score cls  (k,4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]

    labels = valid_mask.nonzero(as_tuple=False)[:, 1]   #k

    if stg == 'std':
        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
            return bboxes, labels

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)    
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]  

        return [dets, labels[keep]]

    else:
        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

            return [[bboxes, labels],[bboxes, labels],[bboxes, labels],[bboxes, labels]]

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)  
        
        res = []
        res.append([bboxes_pre,labels_pre])
        bboxes = torch.cat([bboxes, scores[:, None]], -1)
        res.append([bboxes,labels])
        res.append([dets, labels[keep]])
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num] 
        res.append([dets, labels[keep]])
        return res
