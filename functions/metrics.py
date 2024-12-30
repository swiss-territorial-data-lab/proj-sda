import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_fractional_sets(dets_gdf, labels_gdf, iou_threshold=0.25, area_threshold=None):
    """
    Find the intersecting detections and labels.
    Control their IoU and class to get the TP.
    Tag detections and labels not intersecting or not intersecting enough as FP and FN respectively.
    Save the intersections with mismatched class ids in a separate geodataframe.

    Args:
        dets_gdf (geodataframe): geodataframe of the detections.
        labels_gdf (geodataframe): geodataframe of the labels.
        iou_threshold (float): threshold to apply on the IoU to determine if detections and labels can be matched. Defaults to 0.25.
        area_threshold (float): threshold applied on clipped label and detection polygons to discard the smallest ones. Default None
    Raises:
        Exception: CRS mismatch

    Returns:
        tuple:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detections;
        - geodataframe: false negative labels;
        - geodataframe: intersections between a detection and a label with a mismatched class id.
        - geodataframe: label and detection polygons with an area smaller than the threshold.
        """

    _dets_gdf = dets_gdf.reset_index(drop=True)
    _labels_gdf = labels_gdf.reset_index(drop=True)

    small_poly_gdf = gpd.GeoDataFrame() 
       
    if len(_labels_gdf) == 0:
        fp_gdf = _dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()
        mismatched_classes_gdf = gpd.GeoDataFrame()
        return tp_gdf, fp_gdf, fn_gdf, mismatched_classes_gdf, small_poly_gdf
    
    assert(_dets_gdf.crs == _labels_gdf.crs), f"CRS Mismatch: detections' CRS = {_dets_gdf.crs}, labels' CRS = {_labels_gdf.crs}"

    # we add a id column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf['label_id'] = _labels_gdf.index.astype(int)
    _dets_gdf['det_id'] = _dets_gdf.index.astype(int)
    # We need to keep both geometries after sjoin to check the best intersection over union
    _labels_gdf['label_geom'] = _labels_gdf.geometry

    # Filter detections and labels with area less than a thd value 
    if area_threshold:
        _dets_gdf['area'] = _dets_gdf.area
        filter_dets_gdf = _dets_gdf[_dets_gdf['area']<area_threshold]
        _dets_gdf = _dets_gdf[_dets_gdf['area']>=area_threshold].copy()
        _labels_gdf['area'] = _labels_gdf.area
        filter_labels_gdf = _labels_gdf[_labels_gdf['area']<area_threshold]
        _labels_gdf = _labels_gdf[_labels_gdf['area']>=area_threshold].copy()
        small_poly_gdf = pd.concat([filter_dets_gdf, filter_labels_gdf])

    # TRUE POSITIVES
    left_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')

    # Test that something is detected
    candidates_tp_gdf = candidates_tp_temp_gdf = left_join[left_join.label_id.notnull()].copy()

    # Keep only matching years
    if 'year_label' in candidates_tp_gdf.keys():
        candidates_tp_temp_gdf = candidates_tp_temp_gdf[candidates_tp_temp_gdf.year_label.astype(int) == candidates_tp_temp_gdf.year_det.astype(int)]

    # IoU computation between labels and detections
    geom1 = candidates_tp_temp_gdf['geometry'].to_numpy().tolist()
    geom2 = candidates_tp_temp_gdf['label_geom'].to_numpy().tolist()    
    candidates_tp_temp_gdf.loc[:, ['IOU']] = [intersection_over_union(i, ii) for (i, ii) in zip(geom1, geom2)]

    # Filter detections based on IoU value
    best_matches_gdf = candidates_tp_temp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    best_matches_gdf.drop_duplicates(subset=['det_id'], inplace=True)

    # Detection, resp labels, with IOU lower than threshold value are considered as FP, resp FN, and saved as such
    actual_matches_gdf = best_matches_gdf[best_matches_gdf['IOU'] >= iou_threshold].copy()
    actual_matches_gdf = actual_matches_gdf.sort_values(by=['IOU'], ascending=False).drop_duplicates(subset=['det_id', 'year_det'])
    actual_matches_gdf['IOU'] = actual_matches_gdf.IOU.round(3)

    matched_label_ids = actual_matches_gdf['label_id'].unique().tolist()
    matched_det_ids = actual_matches_gdf['det_id'].unique().tolist()

    fp_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.det_id.isin(matched_det_ids)].drop_duplicates(subset=['det_id'], ignore_index=True)
    fn_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.label_id.isin(matched_label_ids)].drop_duplicates(subset=['label_id'], ignore_index=True)
    fn_gdf_temp.loc[:, 'geometry'] = fn_gdf_temp.label_geom

    # Test that labels and detections share the same class (id starting at 1 for labels and at 0 for detections)
    condition = actual_matches_gdf.label_class == actual_matches_gdf.det_class + 1
    tp_gdf = actual_matches_gdf[condition].reset_index(drop=True)

    mismatched_classes_gdf = actual_matches_gdf[~condition].reset_index(drop=True)
    mismatched_classes_gdf.drop(columns=['x', 'y', 'z', 'dataset_right', 'label_geom'], errors='ignore', inplace=True)
    mismatched_classes_gdf.rename(columns={'dataset_left': 'dataset'}, inplace=True)
  
    # FALSE POSITIVES
    fp_gdf = left_join[left_join.label_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf = pd.concat([fp_gdf_temp, fp_gdf], ignore_index=True)
    fp_gdf.drop(
        columns=_labels_gdf.drop(columns='geometry').columns.to_list() + ['index_right', 'dataset_right', 'label_geom', 'IOU'], 
        errors='ignore', 
        inplace=True
    )
    fp_gdf.rename(columns={'dataset_left': 'dataset'}, inplace=True)

    # FALSE NEGATIVES
    right_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='right', predicate='intersects', lsuffix='left', rsuffix='right')

    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['label_id', 'year_label'], inplace=True)
    fn_gdf = pd.concat([fn_gdf_temp, fn_gdf], ignore_index=True)
    fn_gdf.drop(
        columns=_dets_gdf.drop(columns='geometry').columns.to_list() + ['dataset_left', 'index_right', 'x', 'y', 'z', 'label_geom', 'IOU', 'index_left'], 
        errors='ignore', 
        inplace=True
    )
    fn_gdf.rename(columns={'dataset_right': 'dataset'}, inplace=True)

    return tp_gdf, fp_gdf, fn_gdf, mismatched_classes_gdf, small_poly_gdf


def get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatch_gdf, id_classes=0, method='macro-average'):
    """Determine the metrics based on the TP, FP and FN

    Args:
        tp_gdf (geodataframe): true positive detections
        fp_gdf (geodataframe): false positive detections
        fn_gdf (geodataframe): false negative labels
        mismatch_gdf (geodataframe): labels and detections intersecting with a mismatched class id
        id_classes (list): list of the possible class ids. Defaults to 0.
        method (str): method used to compute multi-class metrics
    
    Returns:
        tuple: 
            - dict: precision for each class
            - dict: recall for each 
            - dict: f1-score for each class
            - float: accuracy
            - float: precision
            - float: recall
            - float: f1 score.
    """

    by_class_dict = {key: 0 for key in id_classes}
    tp_k = by_class_dict.copy()
    fp_k = by_class_dict.copy()
    fn_k = by_class_dict.copy()
    p_k = by_class_dict.copy()
    r_k = by_class_dict.copy()
    f1_k = by_class_dict.copy()
    count_k = by_class_dict.copy()
    pw_k = by_class_dict.copy()
    rw_k = by_class_dict.copy()
 
    for id_cl in id_classes:

        pure_fp_count = 0 if fp_gdf.empty else len(fp_gdf[fp_gdf.det_class==id_cl])
        pure_fn_count = 0 if fn_gdf.empty else len(fn_gdf[fn_gdf.label_class==id_cl+1])  # label class starting at 1 and id class at 0

        mismatched_fp_count = 0 if mismatch_gdf.empty else len(mismatch_gdf[mismatch_gdf.det_class==id_cl])
        mismatched_fn_count = 0 if mismatch_gdf.empty else len(mismatch_gdf[mismatch_gdf.label_class==id_cl+1])

        fp_count = pure_fp_count + mismatched_fp_count
        fn_count = pure_fn_count + mismatched_fn_count
        tp_count = 0 if tp_gdf.empty else len(tp_gdf[tp_gdf.det_class==id_cl])

        tp_k[id_cl] = tp_count
        fp_k[id_cl] = fp_count
        fn_k[id_cl] = fn_count
    
        if tp_count > 0:
            p_k[id_cl] = tp_count / (tp_count + fp_count)
            r_k[id_cl] = tp_count / (tp_count + fn_count)
            f1_k[id_cl] = 2 * p_k[id_cl] * r_k[id_cl] / (p_k[id_cl] + r_k[id_cl])
        count_k[id_cl] = tp_count + fn_count 

    accuracy = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fp_k.values()) + sum(fn_k.values()))

    if method == 'macro-average':   
        precision = sum(p_k.values()) / len(id_classes)
        recall = sum(r_k.values()) / len(id_classes)
    elif method == 'macro-weighted-average': 
        for id_cl in id_classes:
            pw_k[id_cl] = 0 if sum(count_k.values()) == 0 else (count_k[id_cl] / sum(count_k.values())) * p_k[id_cl]
            rw_k[id_cl] = 0 if sum(count_k.values()) == 0 else (count_k[id_cl] / sum(count_k.values())) * r_k[id_cl] 
        precision = sum(pw_k.values()) / len(id_classes)
        recall = sum(rw_k.values()) / len(id_classes)
    elif method == 'micro-average':  
        if sum(tp_k.values()) == 0:
            precision = 0
            recall = 0
        else:
            precision = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fp_k.values()))
            recall = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fn_k.values()))

    if precision==0 and recall==0:
        return tp_k, fp_k, fn_k, p_k, r_k, 0, 0, 0
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return tp_k, fp_k, fn_k, p_k, r_k, f1_k, accuracy, precision, recall, f1


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IOU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    return polygon_intersection / polygon_union


def reliability_diagram(dets_gdf, score='score', output_path='reliability_diagram.jpeg', det_number=True):
    threshold_bins = np.arange(0, 1.05, 0.05)
    bin_values = []
    threshold_values = []
    det_count = []
    for threshold in threshold_bins:
        dets_in_bin = dets_gdf[
            (dets_gdf[score] > threshold-0.5)
            & (dets_gdf[score] <= threshold)
        ]

        if not dets_in_bin.empty:
            det_count.append(dets_in_bin.shape[0])
            bin_values.append(
                dets_in_bin[dets_in_bin.det_category == dets_in_bin.label_category].shape[0] / dets_in_bin.shape[0]
            )
            threshold_values.append(threshold)

    # Make the bin accuracy
    plt.rcParams["figure.figsize"] = (5, 5)
    if det_number:
        fig, ax = plt.subplots(1, 1)

        # Create the barplot
        ax.bar(threshold_values, det_count, alpha=0.5, label='Number of dets', width=0.03)
        ax.set_ylabel('Number of dets in bin')
        # ax.legend(loc='center left')

        # Create a secondary axis
        ax2 = ax.twinx()
    
    else:
        fig, ax2 = plt.subplots(1, 1)

    ax2.scatter(threshold_values, bin_values, marker='o', color='red')
    ax2.plot(threshold_values, bin_values, color='red', label='Detection accuracy')

    ax2.scatter(threshold_bins, threshold_bins, marker='+', color='green')
    ax2.plot(threshold_bins, threshold_bins, color='green', label='Reference line')

    ax2.legend(loc='upper left')

    plt.xlabel(score.replace("_", " "))
    plt.ylabel('bin accuracy')
    plt.title(f'Calibration curve of the {score.replace("_", " ")}')
    plt.grid(True, alpha=0.5)
    
    fig.savefig(output_path, bbox_inches='tight')