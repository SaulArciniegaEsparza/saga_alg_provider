"""
==============================================================================
SAGA GIS algorithm provider for python

Grid Imagery Tools:
    imagery_classification
    imagery_maxent
    imagery_opencv
    imagery_tools
    imagery_vigra


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# env is the provider class
import sys as _sys
import os as _os
import numpy as _np

from ..utilities import files as _files
from ..utilities import data_validation as _validation

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: imagery_classification
# ==============================================================================

def supervised_clasification_grids(classes, grids, training, field=0, quality=None, statistics=None,
                                   normalise=False, load_stats=None, method=2, dist=0., angle=0., prob=0.,
                                   relative_prob=True, wta_ops=None):
    """
    Supervised Classification for grids

    library: imagery_classification  tool: 0

    INPUTS
     classes         [string] output classified grid
     grids           [list, tuple] grids to use for classification analysis
     training        [string] input shapefile with training areas (polygons)
     field           [string, int] training field name or index of input training
                      Selected column must contain class identifier as integers
     quality         [string] optional output grid of distances or probabilities,
                       depending of chosen method
     statistics      [string] optional output statistics file
     normalize       [bool] if True, grids value are normalized
     load_stats      [string] input statistics table
     method          [int] classification method
                      [0] Binary Encoding
                      [1] Parallelepiped
                      [2] Minimum Distance (default)
                      [3] Mahalanobis Distance
                      [4] Maximum Likelihood
                      [5] Spectral Angle Mapping
                      [6] Winner Takes All
     dist            [int, float] threshold distance greater or equal than 0. Let pixel stay unclassified,
                       if minimum euclidian or mahalanobis distance is greater than threshold. If 0 then dist
                       is ignored. Works with method 2 and 3.
     angle           [int, float] Spectral Angle Threshold (Degree) between 0 and 90.
                       Let pixel stay unclassified, if maximum likelihood probability value is less than threshold.
                       Works with method 5.
     prob            [int, float] Probability Threshold between 0 and 100. Let pixel stay unclassified,
                      if maximum likelihood probability value is less than threshold. Works with method 4.
     relative_prob   [bool] if True, relative probability reference is used, in other case, absolute probability
                      is used. Works with method 5.
     wta_ops         [dict] dictionary with Winner Takes All parameters. Next is the example of default options:
                       wta_ops = {'wta_0': False, 'wta_1': False, 'wta_2': False, 'wta_3': False,
                                  'wta_4': False, 'wta_5': False}
    """
    # Inputs and outputs
    cluster = _validation.output_file(classes, 'grid')
    training = _validation.input_file(training, 'vector', True)
    grids = _validation.input_file(grids, 'grid', False)
    grid_list = ';'.join(grids)
    # Optional inputs and outputs
    if quality is None:
        quality = 'NULL'
    else:
        quality = _validation.output_file(quality, 'grid')
    if statistics is not None:
        statistics = _validation.output_file(statistics, 'txt')
    if load_stats is not None:
        load_stats = _validation.input_file(load_stats, 'txt', False)
    # Check parameters
    field, normalise = str(field), str(int(normalise))
    method = _validation.input_parameter(method, 2, vrange=[0, 6], dtypes=[int])
    dist = _validation.input_parameter(dist, 0., gt=0., dtypes=[int, float])
    angle = _validation.input_parameter(angle, 0., vrange=[0., 90.], dtypes=[int, float])
    prob = _validation.input_parameter(prob, 0., vrange=[0., 100.], dtypes=[int, float])
    relative_prob = str(int(relative_prob))
    # Winner Takes All parameters
    wta = ['wta_0', 'wta_1', 'wta_2', 'wta_3', 'wta_4', 'wta_5']
    options = []
    for key in wta:
        options.append('-' + key.upper())
        if type(wta_ops) is dict:
            options.append(str(int(wta_ops.get(key, False))))
        else:
            options.append(str(0))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_classification', '0', '-GRIDS', grid_list,
           '-CLASSES', classes, '-QUALITY', quality, '-TRAINING', training,
           '-TRAINING_CLASS', field, '-METHOD', method, '-THRESHOLD_DIST', dist,
           '-THRESHOLD_ANGLE', angle, '-THRESHOLD_PROB', prob, '-RELATIVE_PROB', relative_prob]
    if statistics is not None:
        cmd.extend(['-FILE_SAVE', statistics])
    if load_stats is not None:
        cmd.extend(['-FILE_LOAD', load_stats])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(grids[0], [cluster])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def kmeans_clustering_grids(cluster, statistics, grids, method=1, ncluster=10,
                            maxiter=0, normalise=False, oldversion=False):
    """
    Cluster Analysis for grids

    library: imagery_classification  tool: 1

    INPUTS
     cluster         [string] output cluster grid
     statistics      [string] output statistics table
     grids           [list, tuple] grids to use for cluster analysis
     method          [int] cluster method
                      [0] Iterative Minimum Distance (Forgy 1965)
                      [1] Hill-Climbing (Rubin 1967) (default)
                      [2] Combined Minimum Distance / Hillclimbing
     ncluster        [int] number of clusters
     maxiter         [int] maximum number of iterations, ignored if set to zero
     normalise       [bool] automatically normalise grids by standard deviation before clustering
     oldversion      [bool] slower but memory saving
    """
    # Inputs and outputs
    cluster = _validation.output_file(cluster, 'grid')
    statistics = _validation.output_file(statistics, 'txt')
    grids = _validation.input_file(grids, 'grid', False)
    grid_list = ';'.join(grids)
    # Check parameters
    method = _validation.input_parameter(method, 1, vrange=[0, 2], dtypes=[int])
    ncluster = _validation.input_parameter(ncluster, 10, gt=2, dtypes=[int])
    maxiter = _validation.input_parameter(maxiter, 0, gt=0, dtypes=[int])
    normalise, oldversion = str(int(normalise)), str(int(oldversion))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_classification', '1', '-GRIDS', grid_list,
           '-CLUSTER', cluster, '-STATISTICS', statistics, '-METHOD', method,
           '-NCLUSTER', ncluster, '-MAXITER', maxiter, '-NORMALISE', normalise,
           '-OLDVERSION', oldversion]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(grids[0], [cluster])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def confusion_matrix_grids(combined, matrix, classval, summary, gridA, gridB, nochange=True,
                           nodata=True, pixels=0, tableA=None, nameA=0, minA=1, maxA=-1,
                           tableB=None, nameB=0, minB=1, maxB=-1):
    """
    Compares two classified grids and creates a confusion matrix and derived coefficients
    as well as the combinations of both classifications as new grid.

    library: imagery_classification  tool: 2

    INPUTS
     combined        [string] output confused grid that combine both input grids
     matrix          [string] output confusion matrix table
     classval        [string] output class values table
     summary         [string] output summary table
     gridA           [string] input grid A
     gridB           [string] input grid B
     nochange        [bool] if True (default) report unchanged classes
     nodata          [bool] if True (default) include unclassified cells (only for SAGA version > 2)
     pixels          [int] consider grid pixels analysis as
                       [0] number of cells (default)
                       [1] percent with respect total area
                       [2] area covered
     tableA          [string] optional classification table file for grid A
     nameA           [str, int] table A field name or index
     minA            [str, int] table A minimum value field name or index
     maxA            [str, int] table A maximum value field name or index. By default is ignored
     tableB          [string] optional classification table file for grid B
     nameB           [str, int] table B field name or index
     minB            [str, int] table B minimum value field name or index
     maxB            [str, int] table B maximum value field name or index. By default is ignored
    """
    # Inputs and outputs
    combined = _validation.output_file(combined, 'grid')
    matrix = _validation.output_file(matrix, 'txt')
    classval = _validation.output_file(classval, 'txt')
    summary = _validation.output_file(summary, 'txt')
    gridA = _validation.input_file(gridA, 'grid', False)
    gridB = _validation.input_file(gridB, 'grid', False)
    if type(tableA) is str:
        tableA = _validation.input_file(tableA, 'txt', False)
    if type(tableB) is str:
        tableB = _validation.input_file(tableB, 'txt', False)
    # Parameters
    nodata, nochange = str(int(nodata)), str(int(nochange))
    pixels = _validation.input_parameter(pixels, 0, vrange=[0, 2], dtypes=[int])
    nameA, minA, maxA = str(nameA), str(minA), str(maxA)
    nameB, minB, maxB = str(nameB), str(minB), str(maxB)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_classification', '2', '-ONE', gridA, '-TWO', gridB,
           '-COMBINED', combined, '-CONFUSION', matrix, '-CLASSES', classval, '-SUMMARY', summary,
           '-NOCHANGE', nochange, '-OUTPUT', pixels]
    if tableA is not None:
        cmd.extend(['-ONE_LUT', tableA, '-ONE_LUT_NAM', nameA, '-ONE_LUT_MIN', minA, '-ONE_LUT_MAX', maxA])
    if tableB is not None:
        cmd.extend(['-TWO_LUT', tableB, '-TWO_LUT_NAM', nameB, '-TWO_LUT_MIN', minB, '-TWO_LUT_MAX', maxB])
    if _env.saga_version[0] != '2':
        cmd.extend(['-NODATA', nodata])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(gridA, combined)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


# ==============================================================================
# Library: imagery_maxent
# ==============================================================================

def maximum_entropy_classification(classes, probability, polygons, field=0,
                                   grids_num=None, grids_cat=None, create_probs=False,
                                   method=0, load_stats=None, save_stats=None,
                                   reg=1, reg_val=1, prob_min=0, nums_real=True,
                                   num_class=32, alpha=0.1, threshold=0.0,
                                   iterations=100):
    """
    Supervised imagery classification using maximum entropy algorithm

    library: imagery_maxent  tool: 0

    INPUTS
        classes          [string] output grid of classes
        probability      [string] output grid of probability
        polygons         [string] input training areas shapefile
        field            [string, int] name or index of the tr
        grids_num        [tuple, list] numerical grids for classification
        grids_cat        [tuple, list] categorical grids for classification
        create_probs     [bool] if True, probability grid for each category
        method           [int] classification method
                          [0] Yoshimasa Tsuruoka (default)
                          [1] Dekang Lin
        load_stats       [string] load statistic file for method=0
        save_stats       [string] save statistic file for method=0
        reg              [int] regularization method for method=0
                          [0] none
                          [1] L1 (default)
                          [2] L2
        reg_val          [int, float] regularization factor for method=0
        prob_min         [int, float] minimum probability
        nums_real        [bool] if True, real-valued numerical features are used.
                          Only for method=0
        num_class        [int] number of numeric value classes
        alpha            [float] alpha factor for method=1
        threshold        [int, float] threshold factor for method=1
        iterations       [int] maximum interations for method=1
    """
    # Inputs and outputs
    classes = _validation.output_file(classes, 'grid')
    probability = _validation.output_file(probability, 'grid')
    polygons = _validation.input_file(polygons, 'vector', False)

    if grids_num is None and grids_cat is None:
        raise TypeError('grids_num and/or grids_class must be input!')
    if type(grids_num) in (list, tuple):
        grids_num = _validation.input_file(grids_num, 'grid', False)
        grids_num_list = ';'.join(grids_num)
    elif grids_num is not None:
        raise TypeError('Bad input type for grids_num <{}>'.format(grids_num))
    if type(grids_cat) in (list, tuple):
        grids_cat = _validation.input_file(grids_cat, 'grid', False)
        grids_cat_list = ';'.join(grids_cat)
    elif grids_cat is not None:
        raise TypeError('Bad input type for grids_class <{}>'.format(grids_num))
    if create_probs:
        out_probs = _os.path.splitext(classes)[0] + '_probs'

    if type(load_stats) is str:
        load_stats = _validation.input_file(load_stats, 'txt', False)
    if type(save_stats) is str:
        save_stats = _validation.output_file(save_stats, 'txt')

    # Parameters
    field = str(field)
    method = _validation.input_parameter(method, 0, vrange=[0, 1], dtypes=[int])
    reg = _validation.input_parameter(reg, 0, vrange=[0, 2], dtypes=[int])
    reg_val, prob_min = str(reg_val), str(prob_min)
    nums_real, num_class = str(int(nums_real)), str(int(num_class))
    alpha, threshold = str(alpha), str(threshold)
    iterations = str(iterations)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_maxent', '0', '-CLASSES', classes,
           '-PROB', probability, '-TRAINING', polygons, '-FIELD', field,
           '-METHOD', method, '-YT_REGUL', reg, '-YT_REGUL_VAL', reg_val,
           '-YT_NUMASREAL', nums_real, '-DL_ALPHA', alpha, '-DL_THRESHOLD',
           threshold, '-DL_ITERATIONS', iterations, '-NUM_CLASSES', num_class,
           '-PROB_MIN', prob_min]
    if grids_num:
        cmd.extend(['-FEATURES_NUM', grids_num_list])
    if grids_cat:
        cmd.extend(['-FEATURES_CAT', grids_cat_list])
    if create_probs:
        cmd.extend(['-PROBS_CREATE', '1', '-PROBS', out_probs])
    if load_stats:
        cmd.extend(['-YT_FILE_LOAD', load_stats])
    if save_stats:
        cmd.extend(['-YT_FILE_SAVE', save_stats])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(polygons, [classes, probability])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def maximum_entropy_prediction(prediction, probability, points, grids_num=None,
                               grids_cat=None, sample_density=1,
                               method=0, load_stats=None, save_stats=None,
                               reg=1, reg_val=1, nums_real=True,
                               num_class=32, alpha=0.1, threshold=0.0,
                               iterations=100):
    """
    Maximum Entropy Presence Prediction

    library: imagery_maxent  tool: 1

    INPUTS
        classes          [string] output grid of presence predictor
        probability      [string] output grid of presence probability
        points           [string] input presence data points
        grids_num        [tuple, list] numerical grids for classification
        grids_cat        [tuple, list] categorical grids for classification
        sample_density   [int, float] background sample density as percent
        method           [int] classification method
                          [0] Yoshimasa Tsuruoka (default)
                          [1] Dekang Lin
        load_stats       [string] load statistic file for method=0
        save_stats       [string] save statistic file for method=0
        reg              [int] regularization method for method=0
                          [0] none
                          [1] L1 (default)
                          [2] L2
        reg_val          [int, float] regularization factor for method=0
        nums_real        [bool] if True, real-valued numerical features are used.
                          Only for method=0
        num_class        [int] number of numeric value classes
        alpha            [float] alpha factor for method=1
        threshold        [int, float] threshold factor for method=1
        iterations       [int] maximum interations for method=1
    """
    # Inputs and outputs
    prediction = _validation.output_file(prediction, 'grid')
    probability = _validation.output_file(probability, 'grid')
    points = _validation.input_file(points, 'vector', False)

    if grids_num is None and grids_cat is None:
        raise TypeError('grids_num and/or grids_class must be input!')
    if type(grids_num) in (list, tuple):
        grids_num = _validation.input_file(grids_num, 'grid', False)
        grids_num_list = ';'.join(grids_num)
    elif grids_num is not None:
        raise TypeError('Bad input type for grids_num <{}>'.format(grids_num))
    if type(grids_cat) in (list, tuple):
        grids_cat = _validation.input_file(grids_cat, 'grid', False)
        grids_cat_list = ';'.join(grids_cat)
    elif grids_cat is not None:
        raise TypeError('Bad input type for grids_class <{}>'.format(grids_num))

    if type(load_stats) is str:
        load_stats = _validation.input_file(load_stats, 'txt', False)
    if type(save_stats) is str:
        save_stats = _validation.output_file(save_stats, 'txt')

    # Parameters
    sample_density = _validation.input_parameter(sample_density, 1, vrange=[0, 100], dtypes=[int, float])
    method = _validation.input_parameter(method, 0, vrange=[0, 1], dtypes=[int])
    reg = _validation.input_parameter(reg, 0, vrange=[0, 2], dtypes=[int])
    reg_val, iterations = str(reg_val), str(iterations)
    nums_real, num_class = str(int(nums_real)), str(int(num_class))
    alpha, threshold = str(alpha), str(threshold)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_maxent', '1', '-PREDICTION', prediction,
           '-PROB', probability, '-PRESENCE', points, '-BACKGROUND', sample_density,
           '-METHOD', method, '-YT_REGUL', reg, '-YT_REGUL_VAL', reg_val,
           '-YT_NUMASREAL', nums_real, '-DL_ALPHA', alpha, '-DL_THRESHOLD',
           threshold, '-DL_ITERATIONS', iterations, '-NUM_CLASSES', num_class]
    if grids_num:
        cmd.extend(['-FEATURES_NUM', grids_num_list])
    if grids_cat:
        cmd.extend(['-FEATURES_CAT', grids_cat_list])
    if load_stats:
        cmd.extend(['-YT_FILE_LOAD', load_stats])
    if save_stats:
        cmd.extend(['-YT_FILE_SAVE', save_stats])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(points, [prediction, probability])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


# ==============================================================================
# Library: imagery_opencv
# ==============================================================================

def opencv_morphological_filter(out_grid, in_grid, method=0, shape=0, radius=1,
                                iterations=1):
    """
    Morphological Filter using OpenCV library

    library: imagery_opencv  tool: 0

    INPUTS
     out_grid       [string] output filtered grid
     in_grid        [string] input grid
     method         [int] operation
                     [0] dilation (default)
                     [1] erosion
                     [2] opening
                     [3] closing
                     [4] morpological gradient
                     [5] top hat
                     [6] black hat
     shape          [int] element shape
                     [0] ellipse (default)
                     [1] rectangle
                     [2] cross
     radius         [int] cell radius
     iterations     [int] number of iterations
    """
    # Inputs and outputs
    out_grid = _validation.output_file(out_grid, 'grid')
    in_grid = _validation.input_file(in_grid, 'grid', False)
    # Parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 6], dtypes=[int])
    shape = _validation.input_parameter(shape, 0, vrange=[0, 2], dtypes=[int])
    radius = _validation.input_parameter(radius, 1, gt=0, dtypes=[int])
    iterations = _validation.input_parameter(iterations, 1, gt=0, dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_opencv', '0', '-INPUT', in_grid, '-OUTPUT',
           out_grid, '-TYPE', method, '-SHAPE', shape, '-RADIUS', radius,
           '-ITERATIONS', iterations]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(in_grid, [out_grid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def opencv_fourier_transform(out_real, out_imag, in_grid):
    """
    Fourier Transformation using OpenCV library

    library: imagery_opencv  tool: 1

    INPUTS
     out_real       [string] output grid of fourier transformation (real)
     out_imag       [string] output grid of fourier transformation (imaginary)
     in_grid        [string] input grid
    """
    # Inputs and outputs
    out_real = _validation.output_file(out_real, 'grid')
    out_imag = _validation.output_file(out_imag, 'grid')
    in_grid = _validation.input_file(in_grid, 'grid', False)
    cmd = ['saga_cmd', '-f=q', 'imagery_opencv', '1', '-INPUT', in_grid,
           '-REAL', out_real, '-IMAG', out_imag]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(in_grid, [out_real, out_imag])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def opencv_normal_bayes_classification(classes, probability, grids, training, field=0,
                                       normalize=False):
    """
    Integration of the OpenCV Machine Learning library for Normal Bayes
    classification of gridded features.

    library: imagery_opencv  tool: 5

    INPUTS
     classes          [string] output grid of presence predictor
     probability      [string] output grid of presence probability
     grids            [tuple, list] grids to use for classification analysis
     training         [string] input training areas shapefile
     field            [int, string] training field name or index of input training
     normalize        [bool] if True, input grids are normalized
    """
    # Inputs and outputs
    classes = _validation.output_file(classes, 'grid')
    probability = _validation.output_file(probability, 'grid')
    training = _validation.input_file(training, 'vector', False)
    if type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('grids must be a tuple or list. < {} > input'.format(type(grids)))
    # Check parameters
    field, normalize = str(field), str(int(normalize))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_opencv', '5', '-FEATURES', grids_list,
           '-NORMALIZE',  normalize, '-PROBABILITY', probability, '-TRAIN_AREAS',
           training, '-TRAIN_CLASS', field, '-CLASSES', classes]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(training, [classes, probability])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def opencv_decision_tree_classification(out_classes, grids, training, field=0, normalize=False,
                                        max_depth=10, min_samples=2, max_categrs=10, use_rule=True,
                                        trunc_pruned=True, reg_acc=0.01):
    """
    Integration of the OpenCV Machine Learning library for Decision Tree
    classification of gridded features.

    library: imagery_opencv  tool: 8

    INPUTS
     out_classes       [string] output classified grid
     grids             [tuple, list] input list of grids to classified
     training          [string] training polygon areas as shapefile
     field             [int, string] index of name of input classification field for
                         training areas
     normalize         [bool] if True, input grids are normalized
     max_depth         [int] maximum tree depth
     min_samples       [int] minimum sample count
     max_categrs       [int] maximum number of categories
     use_rule          [bool] if True (default), use 1SE rule
     trunc_pruned      [bool] if True (default), truncate pruned trees
     reg_acc           [float] regression accuracy
    """
    # Inputs and outputs
    out_classes = _validation.output_file(out_classes, 'grid')
    training = _validation.input_file(training, 'vector', False)
    if type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('grids must be a tuple or list. < {} > input'.format(type(grids)))
    # Parameters
    field, normalize = str(field), str(int(normalize))
    use_rule, trunc_pruned = str(int(use_rule)), str(int(trunc_pruned))
    max_depth = _validation.input_parameter(max_depth, 10, gt=1, dtypes=[int])
    min_samples = _validation.input_parameter(min_samples, 2, gt=2, dtypes=[int])
    max_categrs = _validation.input_parameter(max_categrs, 10, gt=1, dtypes=[int])
    reg_acc = _validation.input_parameter(reg_acc, 0.01, gt=0, dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_opencv', '8', '-CLASSES', out_classes,
           '-FEATURES', grids_list, '-TRAIN_AREAS', training, '-TRAIN_CLASS',
           field, '-NORMALIZE', normalize, '-MAX_DEPTH', max_depth, '-MIN_SAMPLES',
           min_samples, '-MAX_CATEGRS', max_categrs, '-1SE_RULE', use_rule,
           '-TRUNC_PRUNED', trunc_pruned, '-REG_ACCURACY', reg_acc]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(training, [out_classes])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def opencv_boosting_classification(out_classes, grids, training, field=0, normalize=False, method=1,
                                   max_depth=10, min_samples=2, max_categrs=10, use_rule=True,
                                   trunc_pruned=True, reg_acc=0.01, weak_count=100, trim_rate=0.95):
    """
    Integration of the OpenCV Machine Learning library for Boosted Trees
    classification of gridded features

    library: imagery_opencv  tool: 9

    INPUTS
     out_classes       [string] output classified grid
     grids             [tuple, list] input list of grids to classified
     training          [string] training polygon areas as shapefile
     field             [int, string] index of name of input classification field for
                         training areas
     normalize         [bool] if True, input grids are normalized
     method            [int] Boost Type
                        [0] Discrete AdaBoost
                        [1] Real AdaBoost (default)
                        [2] LogitBoost
                        [3] Gentle AdaBoost
     max_depth         [int] maximum tree depth
     min_samples       [int] minimum sample count
     max_categrs       [int] maximum number of categories
     use_rule          [bool] if True (default), use 1SE rule
     trunc_pruned      [bool] if True (default), truncate pruned trees
     reg_acc           [float] regression accuracy
     weak_count        [int] weak count
     trim_rate         [float] weight trim rate
    """
    # Inputs and outputs
    out_classes = _validation.output_file(out_classes, 'grid')
    training = _validation.input_file(training, 'vector', False)
    if type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('grids must be a tuple or list. < {} > input'.format(type(grids)))
    # Parameters
    field, normalize = str(field), str(int(normalize))
    use_rule, trunc_pruned = str(int(use_rule)), str(int(trunc_pruned))
    method = _validation.input_parameter(method, 1, vrange=[0, 3], dtypes=[int])
    max_depth = _validation.input_parameter(max_depth, 10, gt=1, dtypes=[int])
    min_samples = _validation.input_parameter(min_samples, 2, gt=2, dtypes=[int])
    max_categrs = _validation.input_parameter(max_categrs, 10, gt=1, dtypes=[int])
    reg_acc = _validation.input_parameter(reg_acc, 0.01, gt=0, dtypes=[float])
    weak_count = _validation.input_parameter(weak_count, 100, gt=0, dtypes=[int])
    trim_rate = _validation.input_parameter(trim_rate, 0.95, vrange=[0, 1], dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_opencv', '9', '-CLASSES', out_classes,
           '-FEATURES', grids_list, '-TRAIN_AREAS', training, '-TRAIN_CLASS',
           field, '-NORMALIZE', normalize, '-BOOST_TYPE', method, '-MAX_DEPTH',
           max_depth, '-MIN_SAMPLES', min_samples, '-MAX_CATEGRS', max_categrs,
           '-1SE_RULE', use_rule, '-TRUNC_PRUNED', trunc_pruned, '-REG_ACCURACY',
           reg_acc, '-WEAK_COUNT', weak_count, '-WGT_TRIM_RATE', trim_rate]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(training, [out_classes])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


# ==============================================================================
# Library: imagery_tools
# ==============================================================================

def vegetation_index_db(basename, red, nir, intercept=0, slope=0.5):
    """
    Distance based vegetation indices.

    library: imagery_tools  tool: 0

    INPUTS
        basename      [string] output basename file. Several outputs will be created:
                        '_PVI0'   Perpendicular Vegetation Index (Richardson and Wiegand, 1977)
                        '_PVI1'   Perpendicular Vegetation Index (Perry and Lautenschlager, 1984)
                        '_PVI2'   Perpendicular Vegetation Index (Walther and Shabaani)
                        '_PVI3'   Perpendicular Vegetation Index (Qi, et al., 1994)
                        '_TSAVI'  Transformed Soil Adjusted Vegetation Index (Baret et al. 1989)
                        '_ATSAVI' Transformed Soil Adjusted Vegetation Index (Baret and Guyot, 1991)
        red           [string] red reflectance input grid
        nir           [string] near infrared reflectance input grid
        intercept     [float] intercept of soil line
        slope         [float] slope of soil line
    """
    # Inputs and outputs
    var_ext = ['PVI0', 'PVI1', 'PVI2', 'PVI3', 'TSAVI', 'ATSAVI']
    outgrids = [_validation.output_file(basename + '_' + fext, 'grid') for fext in var_ext]

    red = _validation.input_file(red, 'grid', False)
    nir = _validation.input_file(nir, 'grid', False)
    # Validate parameters
    intercept, slope = str(intercept), str(slope)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_tools', '0', '-RED', red, '-NIR', nir,
           '-INTERCEPT', intercept, '-SLOPE', slope]
    for i, key in enumerate(var_ext):
        cmd.extend(['-' + key, outgrids[i]])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(red, outgrids)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vegetation_index_sb(basename, red, nir, factor=0.5):
    """
    Slope based vegetation indices.

    library: imagery_tools  tool: 1

    INPUTS
        basename      [string] output basename file. Several outputs will be created:
                        '_DVI'  Difference Vegetation Index DVI = NIR - R
                        '_NDVI' Normalized Difference Vegetation Index NDVI = (NIR - R) / (NIR + R)
                        '_RVI'  Ratio Vegetation Index RVI = R / NIR
                        '_NRVI' Normalized Ratio Vegetation Index NRVI = (RVI - 1) / (RVI + 1)
                        '_TVI'  Transformed Vegetation Index TVI = [(NIR - R) / (NIR + R) + 0.5]^0.5
                        '_CTVI' Corrected Transformed Ratio Vegetation Index
                                CTVI = [(NDVI + 0.5) / abs(NDVI + 0.5)] * [abs(NDVI + 0.5)]^0.5
                        '_TTVI' Thiam's Transformed Vegetation Index RVI = [abs(NDVI) + 0.5]^0.5
                        '_SAVI' Soil Adjusted Vegetation Index SAVI = [(NIR - R) / (NIR + R)] * (1 + S)
        red           [string] red reflectance input grid
        nir           [string] near infrared reflectance input grid
        factor        [float] Soil adjustment factor for SAVI calculation.
                              Suggested values are 1.0 in case of very low vegetation,
                              0.5 for intermediate 0.5, and 0.25 for high densities (Silleos et al. 2006).
    """
    # Inputs and outputs
    var_ext = ['DVI', 'NDVI', 'RVI', 'NRVI', 'TVI', 'CTVI', 'TTVI', 'SAVI']
    outgrids = [_validation.output_file(basename + '_' + fext, 'grid') for fext in var_ext]

    red = _validation.input_file(red, 'grid', False)
    nir = _validation.input_file(nir, 'grid', False)
    # Validate parameters
    factor = _validation.input_parameter(factor, 0.5, vrange=[0, 1], dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_tools', '1', '-RED', red, '-NIR', nir,
           '-SOIL', factor]
    for i, key in enumerate(var_ext):
        cmd.extend(['-' + key, outgrids[i]])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(red, outgrids)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def enhanced_vegetation_index(evi, blue, red, nir, gain=2.5, l=1.0,
                              cblue=7.5, cred=6.0):
    """
    Enhanced Vegetation Index (EVI) from Huete et al (2002)

    library: imagery_tools  tool: 2

    INPUTS
        evi      [string] output evi grid
        blue     [string] input blue reflectance grid
        red      [string] input red reflectance grid
        nir      [string] input nir reflectance grid
        gain     [float] gain factor
        l        [float] canopy background adjustment
        cblue    [float] aerosol resistance coefficient (blue)
        cred     [float] aerosol resistance coefficient (red)
    """
    # Inputs and outputs
    evi = _validation.output_file(evi, 'grid')
    blue = _validation.input_file(blue, 'grid', False)
    red = _validation.input_file(red, 'grid', False)
    nir = _validation.input_file(nir, 'grid', False)
    # Parameters
    gain = _validation.input_parameter(gain, 2.5, gt=0., dtypes=[float])
    l = _validation.input_parameter(l, 1.0, gt=0., dtypes=[float])
    cblue = _validation.input_parameter(cblue, 7.5, gt=0., dtypes=[float])
    cred = _validation.input_parameter(cred, 6.0, gt=0., dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_tools', '2', '-BLUE', blue, '-RED', red,
           '-NIR', nir, '-EVI', evi, '-GAIN', gain, '-L', l, '-CBLUE', cblue,
           '-CRED', cred]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(blue, [evi])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def ihs_sharpening(out_red, out_green, out_blue, in_red, in_green, in_blue, pan,
                   resampling=2, pan_match=0):
    """
    Intensity, hue, saturation (IHS) sharpening.

    library: imagery_tools  tool: 4

    INPUTS
        out_red       [string] output sharped red reflectance grid
        out_green     [string] output sharped green reflectance grid
        out_blue      [string] output sharped blue reflectance grid
        in_red        [string] input red reflectance grid
        in_green      [string] input green reflectance grid
        in_blue       [string] input blue reflectance grid
        pan           [string, list, tuple] input panchromatic channel grid or list of grids
        resampling    [int] resampling method
                       [0] nearest neighbour
                       [1] bilinear
                       [2] cubic convolution (default)
        pan_match     [int] panchromatic channel matching method
                       [0] normalized (default)
                       [1] standardized
    """
    # Inputs and outputs
    out_red = _validation.output_file(out_red, 'grid')
    out_green = _validation.output_file(out_green, 'grid')
    out_blue = _validation.output_file(out_blue, 'grid')
    in_red = _validation.input_file(in_red, 'grid', False)
    in_green = _validation.input_file(in_green, 'grid', False)
    in_blue = _validation.input_file(in_blue, 'grid', False)
    if type(pan) is str:
        pan_list = _validation.input_file(pan, 'grid', False)
    elif type(pan) in (list, tuple):
        pan = [_validation.input_file(name, 'grid', False) for name in pan]
        pan_list = ",".join(pan)
    else:
        raise TypeError('pan must be a string file name or tuple/list of files')
    # Validate parameters
    resampling = _validation.input_parameter(resampling, 2, vrange=[0, 2], dtypes=[int])
    pan_match = _validation.input_parameter(pan_match, 0.1, vrange=[0, 1], dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '4', '-R', in_red, '-G', in_green, '-B', in_blue,
           '-PAN', pan_list, '-R_SHARP', out_red, '-G_SHARP', out_green, '-B_SHARP', out_blue,
           '-RESAMPLING', resampling, '-PAN_MATCH', pan_match]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(in_red, [out_blue, out_green, out_red])

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def principle_components(outgrids, pan, grids, method=1, resampling=2, pan_match=1):
    """
    Principle components based image sharpening.

    library: imagery_tools  tool: 7

    INPUTS
        outgrids      [list, tuple] name of the output grids, one for each input grid
        pan           [string] panchromatic channel input grid
        grids         [list, tuple] name of input grids
        method        [int] principle components method
                       [0] correlation matrix
                       [1] variance-covariance matrix (default)
                       [2] sums-of-squares-and-cross-products matrix
        resampling    [int] resampling method
                       [0] nearest neighbour
                       [1] bilinear
                       [2] cubic convolution (default)
        pan_match     [int] panchromatic channel matching method
                       [0] normalized
                       [1] standardized (default)
    """
    # Inputs and outputs
    pan = _validation.input_file(pan, 'grid', False)
    if type(outgrids) in (list, tuple):
        outgrids = [_validation.output_file(name, 'grid') for name in outgrids]
        outgrids_list = ",".join(outgrids)
    else:
        raise TypeError('outgrids must be a tuple/list of files')
    if type(grids) in (list, tuple):
        grids = [_validation.output_file(name, 'grid') for name in grids]
        grids_list = ",".join(grids)
    else:
        raise TypeError('grids must be a tuple/list of files')
    if len(outgrids_list) != len(grids_list):
        raise ValueError('outgrids and grids must have the same length!')
    # Validate parameters
    method = _validation.input_parameter(method, 1, vrange=[0, 2], dtypes=[int])
    resampling = _validation.input_parameter(resampling, 2, vrange=[0, 2], dtypes=[int])
    pan_match = _validation.input_parameter(pan_match, 2, vrange=[0, 1], dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_tools', '7', '-GRIDS', grids_list, '-PAN', pan,
           '-SHARPEN', outgrids_list, '-RESAMPLING', resampling, '-PAN_MATCH', pan_match, '-METHOD', method]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(pan, outgrids)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


# ==============================================================================
# Library: imagery_vigra
# ==============================================================================

def vigra_smoothing(outgrid, ingrid, method=0, scale=2, threshold=1):
    """
    Smoothing (ViGrA)

    library: imagery_vigra  tool: 0

    INPUTS
        outgrid      [string] output smoothed grid
        ingrid       [string] input grid
        method       [int] filter type
                      [0] exponential (default)
                      [1] nonlinear
                      [2] gaussian
        scale        [int] size of smoothing filter
        threshold    [float] edge threshold for nonlinear smoothing
    """
    # Inputs and outputs
    outgrid = _validation.output_file(outgrid, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Validate parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 2], dtypes=[int])
    scale, threshold = str(scale), str(threshold)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '0', '-OUTPUT', outgrid, '-INPUT', ingrid,
           '-TYPE', method, '-SCALE', scale, '-EDGE', threshold]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(ingrid, outgrid)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vigra_edge_detection(outgrid, ingrid, method=0, scale=1, threshold=1):
    """
    Edge Detection (ViGrA)

    library: imagery_vigra  tool: 1

    INPUTS
        outgrid      [string] output edge grid file
        ingrid       [string] input grid
        method       [int] edge detector type
                      [0] Canny (default)
                      [1] Shen-Castan
        scale        [int, float] Operator scale
        threshold    [int, float] Gradient threshold
    """
    # Inputs and outputs
    outgrid = _validation.output_file(outgrid, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Validate parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 1], dtypes=[int])
    scale, threshold = str(scale), str(threshold)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '1', '-INPUT', ingrid, '-OUTPUT', outgrid,
           '-TYPE', method, '-SCALE', scale, '-THRESHOLD', threshold]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(ingrid, outgrid)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vigra_morphological_filter(outgrid, ingrid, method=0, radius=1, rank=0.5, rescale=True):
    """
    Morphological Filter (ViGrA)

    library: imagery_vigra  tool: 2

    INPUTS
        outgrid      [string] output filtered grid
        ingrid       [string] input grid
        method       [int] filter type
                      [0] Dilation (default)
                      [1] Erosion
                      [2] Median
                      [3] User defined rank
        radius       [int] radius as number of cells
        rank         [float] user defined rank for method=3
        rescale      [bool] rescale values between 0-255. By default True
    """
    # Inputs and outputs
    outgrid = _validation.output_file(outgrid, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Validate parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 3], dtypes=[int])
    rank = _validation.input_parameter(rank, 0.5, vrange=[0, 1], dtypes=[float])
    radius, rescale = str(int(radius)), str(int(rescale))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '2', '-OUTPUT', outgrid, '-INPUT', ingrid,
           '-TYPE', method, '-RADIUS', radius, '-RANK', rank, '-RESCALE', rescale]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(ingrid, outgrid)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vigra_fourier_transform(out_real, out_imag, ingrid, center=True):
    """
    Fourier Transform (ViGrA)

    library: imagery_vigra  tool: 5

    INPUTS
        out_real     [string] output real grid
        out_imag     [string] output imaginary grid
        ingrid       [string] input grid
        center       [bool] centered image. By default True
    """
    # Inputs and outputs
    out_real = _validation.output_file(out_real, 'grid')
    out_imag = _validation.output_file(out_imag, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Validate parameters
    center = str(int(center))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '5', '-INPUT', ingrid, '-REAL=', out_real,
           '-IMAG', out_imag, '-CENTER', center]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(ingrid, [out_imag, out_real])

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vigra_fourier_transform_inv(outgrid, in_real, in_imag, center=True):
    """
    Fourier Transform Inverse (ViGrA)

    library: imagery_vigra  tool: 6

    INPUTS
        outgrid      [string] output grid
        in_real      [string] input real grid
        in_imag      [string] input imaginary grid
        center       [bool] centered image. By default True
    """
    # Inputs and outputs
    outgrid = _validation.output_file(outgrid, 'grid')
    in_real = _validation.input_file(in_real, 'grid', False)
    in_imag = _validation.input_file(in_imag, 'grid', False)
    # Validate parameters
    center = str(int(center))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '6', '-OUTPUT', outgrid, '-REAL=', in_real,
           '-IMAG', in_imag, '-CENTER', center]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(in_real, outgrid)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def vigra_fourier_filter(outgrid, ingrid, method=0, scale=2, power=0.5, vmin=0.1, vmax=0.9):
    """
    Fourier Filter (ViGrA)

    library: imagery_vigra  tool: 8

    INPUTS
        outgrid      [string] output edge grid file
        ingrid       [string] input grid
        method       [int] filter type
                      [0] gaussian (default)
                      [1] power of distance
                      [2] include range
                      [3] exclude range
        scale        [int, float] size of smoothing filter. Only for method=0
        power        [int, float] power factor for method=1
        vmin         [int, float] minimum value for method 2 and 3
        vmax         [int, float] maximum value for method 2 and 3
    """
    # Inputs and outputs
    outgrid = _validation.output_file(outgrid, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Validate parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 3], dtypes=[int])
    vmin = _validation.input_parameter(vmin, 0.1, vrange=[0, 1], dtypes=[float])
    vmax = _validation.input_parameter(vmax, 0.9, vrange=[0, 1], dtypes=[float])
    scale, power = str(scale), str(power)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_vigra', '8', '-INPUT', ingrid, '-OUTPUT', outgrid,
           '-SCALE', scale, '-POWER', power, '-RANGE_MIN', vmin, '-RANGE_MAX', vmax,
           '-FILTER', method]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(ingrid, outgrid)

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))

