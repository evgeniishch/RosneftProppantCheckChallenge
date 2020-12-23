import numpy as np
import pandas as pd

sieves_names = [6,7,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,100,'pan']
sieves_names = [str(i) for i in sieves_names]

sive_diam = [3.35,2.8, 2.36, 2, 1.7, 1.4, 1.18, 1, 0.85, 0.71, 0.6, 0.5, 0.425, 0.355, 0.3, 0.25, 0.212, 0.18, 0.15]
sive_diam = np.array(sive_diam)

sieves_names_pan = [6,7,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,100,0]
sive_diam_pan = np.array([3.35,2.8, 2.36, 2, 1.7, 1.4, 1.18, 1, 0.85, 0.71, 0.6, 0.5, 0.425, 0.355, 0.3, 0.25, 0.212, 0.18, 0.15,0])
#diams_exended = np.concatenate([[3.35],sive_diam])

fraction_sievs = {
        '16/20' : {'all' : ['12', '16', '18', '20', '25', '30', '40'], 'main': ['18', '20'], 'rough':'12' },
        '20/40' : {'all' : ['16', '20', '25', '30', '35', '40', '50'], 'main': ['25', '30', '35', '40'], 'rough':'16' },
        '20/40_pdcpd_bash_lab' : {'all' : ['12', '16', '18', '20', '25', '30', '40'], 'main': ['25', '30', '40'], 'rough':'12' }
}

true_cols = ['6_true','7_true', '8_true',
           '10_true', '12_true', '14_true', '16_true', '18_true', '20_true',
           '25_true', '30_true', '35_true', '40_true', '45_true', '50_true',
           '60_true', '70_true', '80_true', '100_true', 'pan_true']
pred_cols = ['6_pred', '7_pred', '8_pred', '10_pred', '12_pred', '14_pred', '16_pred',
           '18_pred', '20_pred', '25_pred', '30_pred', '35_pred', '40_pred',
           '45_pred', '50_pred', '60_pred', '70_pred', '80_pred', '100_pred','pan_pred']
cols =  ['6', '7', '8', '10', '12', '14', '16', '18', '20', '25', '30',
       '35', '40', '45', '50', '60', '70', '80', '100', 'pan']


def contest_metric(true_labels, submission):
    """
    Полный пайплайн расчета метрики.
    Аргументы:
    true_labels - pd.DataFrame с true разметкой 
    submission - pd.DataFrame содержащий размеры гранул

    Вернёт:
    Значение итоговой метрики (одно число)
    """
    sub_prep = prepare_sieves_hist_df(submission, true_labels)
    prop_count_df = prepare_prop_count_df(submission, true_labels)
    return metric(sub_prep, prop_count_df)



def metric(bin_hist_sub_df, prop_count_sub_df, a=0.6, b=0.4):
    """
    Расчёт итоговой метрики.
    Аргументы:
    bin_hist_sub_df - pd.DataFrame содержащий true и pred колонки по каждому бину (ситам)
    prop_count_sub_df - pd.DataFrame содержащий true и pred колонки по количеству гранул
    a - коэффициент для ошибки совпадения гистограмм (chi2)
    b - коэффициент для ошибки определения количества гранул (mape)
    Вернёт:
    Значение итоговой метрики (одно число)
    """
    bin_hist_sub_df['chi2'] = bin_hist_sub_df.apply(lambda x: calc_chi_square_metric(x[true_cols].values,
                                                                    x[pred_cols].values,
                                                                    x['fraction']), axis=1)
    chi2 = bin_hist_sub_df['chi2'].mean()
    prop_count_sub_df['mape'] = np.abs(prop_count_sub_df['prop_count_true'] - prop_count_sub_df['prop_count_pred']) / prop_count_sub_df['prop_count_true']
    mape = prop_count_sub_df['mape'].mean()
    return a*chi2 + b*mape, chi2, mape

def get_bins_from_granules(x):
    """Развернёт бины для каждого изображения по столбцам"""
    res_dict = sizes_to_sieve_hist(x['prop_size'].values,
                                    sive_diam_pan,
                                    sieves_names_pan)
    ser = pd.Series(res_dict)
    return ser

def prepare_prop_count_df(submission_df, test_labels_df):
    """
    Подготовит датафрейм для подсчёта ошибки определения количества гранул.
    Аргументы:
    submission_df - pd.DataFrame в формате ImageId | prop_size
    test_labels_df - pd.DataFrame содержащий истинные метки для распределений по ситам и количеству гранул
    Вернёт:
    pd.DataFrame содержащий true и pred колонки по количеству гранул
    """
    prop_count_pred = pd.DataFrame(submission_df.groupby('ImageId')['prop_size'].count())
    prop_count_pred.reset_index(inplace=True)
    prop_count_pred.columns = ['ImageId','prop_count_pred']
    prop_count_true = test_labels_df[~test_labels_df['prop_count'].isna()][['ImageId', 'prop_count']].copy()
    prop_count_true.columns = ['ImageId', 'prop_count_true']
    prop_count_df = pd.merge(prop_count_pred, prop_count_true, on='ImageId')
    return prop_count_df

def prepare_sieves_hist_df(submission_df, test_labels_df_in, result_bins=None):
    """
    Подготовит датафрейм для подсчёта ошибки совпадения гистограмм.
    Аргументы:
    submission_df - pd.DataFrame в формате ImageId | prop_size
    test_labels_df - pd.DataFrame содержащий истинные метки для распределений по ситам и количеству гранул
    Вернёт:
    pd.DataFrame содержащий true и pred колонки по каждому бину (ситам)
    """
    test_labels_df = test_labels_df_in.copy()
    if result_bins is None:
        result_bins = pd.DataFrame(submission_df.groupby('ImageId').apply(lambda x: get_bins_from_granules(x)))
        result_bins.rename(lambda x: str(x) + '_pred', axis='columns',inplace=True)
        result_bins.reset_index(inplace=True)
    test_labels_df.rename(lambda x: str(x) + '_true' if x in cols else x, axis='columns', inplace=True)

    sieves_hist_df = pd.merge(result_bins,test_labels_df, on='ImageId')
    sieves_hist_df = sieves_hist_df[~sieves_hist_df['pan_true'].isna()].drop('prop_count', axis=1).copy()
    return sieves_hist_df

def calc_chi_square_on_sizes(sizes, true_hist, fraction):
    pred_hist = sizes_to_sieves(sizes, sive_diam, sieves_names)
    return calc_chi_square_metric(true_hist, pred_hist, fraction)

def chi_square_metric(true, pred):
    """
    Возвращает значение метрики хи-квадрат на прогнозе и реальном значении
    """
    mask = ((true==0.0) & (pred==0.0)) | pd.isna(true) | pd.isna(pred)
    true = true[mask==False]
    pred = pred[mask==False]
    distances = ((true-pred)**2)/(true+pred)
    distances[(true==0.0) & (pred==0.0) | (pd.isna(distances))] = 0.0
    return np.sum(distances)/2

def postprocess_dists(true, pred, sieve_mask):
    true = np.append(true, 0.0)
    pred = np.append(pred, 0.0)

    non_zero_bins = np.argwhere(sieve_mask)
    non_zero_bins = np.insert(non_zero_bins, 0 ,-1, axis=0)
    
    for i, j in zip(non_zero_bins[:-1], non_zero_bins[1:]):
        pred[j] = np.sum(pred[i[0]+1:j[0]+1])

    pred[-1] = np.sum(pred[non_zero_bins[-1][0]+1: ])

    pred[np.argwhere(sieve_mask==0)[:-1]] = 0.0
    true[-1] = np.maximum(0.0, 1.0 - np.sum(true[:-1]))
    return true, pred

def calc_chi_square_metric(true_hist, pred_hist, fraction):
    sieve_mask = np.array([1 if x in fraction_sievs[fraction]['all'] else 0 for x in sieves_names])
    true_hist_processed, pred_hist_processed = postprocess_dists(true_hist, pred_hist, sieve_mask)
    return chi_square_metric(true_hist_processed, pred_hist_processed)

def ret_sieve_mask(fraction):
    return np.array([1 if x in fraction_sievs[fraction]['all'] else 0 for x in sieves_names])

def sizes_to_sieves(sizes, sive_diam, sieves_names):
    """
    Распределяет предикты по ситам
    """
    sizes_ = np.sort(sizes)
    sieve_bins = np.zeros_like(sizes_)

    for diam, name in zip(sive_diam, sieves_names):
        sieve_bins[sizes_<= diam] = name
        
    return sizes_, sieve_bins

def sizes_to_sieve_hist(sizes, sive_diam, sieves_names):
    """
    Считает бины сит
    """
    sizes_, sieve_bins = sizes_to_sieves(sizes, sive_diam, sieves_names)
    bins_hieght = dict()
    
    for name in sieves_names:
        bins_hieght[name] = np.sum(sieve_bins==float(name))/sizes.shape[0]
    bins_hieght['pan'] = bins_hieght.pop(0)
    return bins_hieght

