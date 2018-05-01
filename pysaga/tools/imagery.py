"""
SAGA GIS algorithm provider
Grids tools:
    grid_analysis
    grid_calculus
    grid_filter
    grid_gridding
    grid_spline
    grid_tools


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# env is the provider class
import sys as _sys
import os as _os
import numpy as _np

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: imagery_classification
# ==============================================================================

def supervised_clasification_grids(classes, grids, polygons, field=0, quality=None, statistics=None,
                                   normalise=False, load_stats=None, method=2, dist=0., angle=0., prob=0.,
                                   relative_prob=True, wta_ops=None):
    """
    Supervised Classification for grids

    library: imagery_classification  tool: 0

    INPUTS
     classes         [string] output classified grid
     grids           [list, tuple] grids to use for classification analysis
     polygons        [string] input shapefile with training areas (polygons)
     field           [string, int] training field name or index of input polygons.
                      Selected column must contain class identifier as integers.
     quality         [string] optional output grid of distances or probabilities,
                       depending of chosen method.
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
    polygons = _validation.input_file(polygons, 'vector', True)
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
           '-CLASSES', classes, '-QUALITY', quality, '-TRAINING', polygons,
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

def maximum_entropy_classification(out_classes, out_prob, training, field=0, grids_num=None,
                                   grids_cat=None, create_probs=False, method=0, load_stats=None,
                                   statistics=None, regul_method=1, regul_val=1., num_real=True,
                                   min_prob=0, alpha=0.1, threshold=0., iterations=100, classes=32):
    # Inputs and outputs
    out_classes = _validation.output_file(out_classes, 'grid')
    out_prob = _validation.output_file(out_prob, 'grid')
    training = _validation.output_file(training, 'vector')
    if type(grids_num) in (list, tuple):
        grids_num = _validation.input_file(grids_num, 'grid', False)
        grids_num_list = ';'.join(grids_num)
    elif grids_num is None:
        pass
    else:
        raise TypeError('grids_num must be a tuple or list. < {} > input'.format(type(grids_num)))
    if type(grids_cat) in (list, tuple):
        grids_cat = _validation.input_file(grids_cat, 'grid', False)
        grids_cat_list = ';'.join(grids_cat)
    elif grids_num is None:
        pass
    else:
        raise TypeError('grids_cat must be a tuple or list. < {} > input'.format(type(grids_cat)))
    if statistics is not None:
        statistics = _validation.output_file(statistics, 'txt')
    if load_stats is not None:
        load_stats = _validation.input_file(load_stats, 'txt', False)
    # Parameters
    field = str(field)
    create_probs, num_real = str(int(create_probs)), str(int(num_real))
    method = _validation.input_parameter(method, 0, vrange=[0, 1], dtypes=[int])
    regul_method = _validation.input_parameter(regul_method, 0, vrange=[0, 1], dtypes=[int])
    regul_val = _validation.input_parameter(regul_val, 1., gt=0., dtypes=[int, float])
    alpha = _validation.input_parameter(alpha, 0.1, gt=0., dtypes=[float])
    threshold = _validation.input_parameter(threshold, 0., gt=0., dtypes=[float])
    iterations = _validation.input_parameter(iterations, 100, gt=1, dtypes=[int])
    classes = _validation.input_parameter(classes, 32, gt=1, dtypes=[int])
    min_prob = _validation.input_parameter(min_prob, 0., vrange=[0, 1], dtypes=[float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'imagery_maxent', '0', '-TRAINING', training, '-FIELD', field,
           '-CLASSES', out_classes, '-PROB', out_prob, '-PROBS_CREATE', create_probs, '-METHOD',
           method, '-YT_REGUL', regul_method, '-YT_REGUL_VAL', regul_val, '-YT_NUMASREAL', num_real,
           '-DL_ALPHA', alpha, '-DL_THRESHOLD', threshold, '-DL_ITERATIONS', iterations, '-NUM_CLASSES',
           classes, '-PROB_MIN', min_prob]
    if type(grids_num) is not None:
        cmd.extend(['-FEATURES_NUM', grids_num_list])
    if type(grids_cat) is not None:
        cmd.extend(['-FEATURES_CAT', grids_cat_list])
    if statistics is not None:
        cmd.extend(['-YT_FILE_SAVE', statistics])
    if load_stats is not None:
        cmd.extend(['-YT_FILE_LOAD', load_stats])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(grids[0], [cluster])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


# ==============================================================================
# Library: imagery_opencv
# ==============================================================================


def opencv_boosting_classification(out_classes, grids, training, field=0, normalize=False, method=1,
                                   max_depth=10, min_samples=2, max_categrs=10, use_rule=True,
                                   trunc_pruned=True, reg_acc=0.01, weak_count=100, trim_rate=0.95):
    """
    Compares two classified grids and creates a confusion matrix and derived coefficients
    as well as the combinations of both classifications as new grid.

    library: imagery_classification  tool: 2

    INPUTS
     combined        [string] output confused grid that combine both input grids
    """
    # Inputs and outputs
    out_classes = _validation.output_file(out_classes, 'grid')
    training = _validation.output_file(training, 'vector')
    if type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('grids must be a tuple or list. < {} > input'.format(type(grids_num)))
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
    _validation.validate_crs(grids[0], [out_classes])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def opencv_decision_tree_classification():
    pass


def opencv_fourier_transform():
    pass


def opencv_normal_bayes_classification():
    pass


def opencv_morphological_filter():
    pass


# ==============================================================================
# Library: imagery_tools
# ==============================================================================


def enhanced_vegetation_index():
    pass


def principle_components():
    pass


def vegetation_index():
    pass


# ==============================================================================
# Library: imagery_vigra
# ==============================================================================

def vigra_edge_detection():
    pass


def vigra_fourier_filter():
    pass


def vigra_fourier_transform():
    pass


def vigra_fourier_transform_inv():
    pass


def vigra_morphological_filter():
    pass


def vigra_smoothing():
    pass


