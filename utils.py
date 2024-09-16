import os
import numpy as np
import pandas as pd
from FixedPointFinderTF2 import FixedPointFinderTF2
from FixedPoints import FixedPoints


def optimization_wave(idp, all_initial_states, FPS_DIR, file_text="", cond_ids=None, SAVE=False,
                      **config_optim):
 
    if config_optim['n_chunks'] != 1:
        inputs = np.zeros((all_initial_states.shape[0], idp.embedding_dim))
        fpf = FixedPointFinderTF2(idp.rnn_cell, lr_init=config_optim['lr_init'],
                                  max_iters=config_optim['max_iters'],
                                  do_exclude_distance_outliers = config_optim['exclude_outliers'],
                                  method='joint', verbose=1)
        print(f"le he dicho que: {fpf.do_exclude_distance_outliers}")
        _, all_fps = fpf.find_fixed_points(all_initial_states, inputs, cond_ids)
        all_fps = all_fps.sort('qstar')
        all_initial_states = all_fps.x_init

    alls = []
    uniques = []
    for i in range(config_optim['n_chunks']):

        print(f"para el bloque {i + 1}")

        initial_states = all_initial_states[
                         config_optim['states_per_chunk'] * i:config_optim['states_per_chunk'] * (i + 1)]
        inputs = np.zeros((initial_states.shape[0], idp.embedding_dim))

        if config_optim['noise_sigma'] > 0:
            mu, sigma = 0, config_optim['noise_sigma']
            a, b = initial_states.shape
            s = np.random.normal(mu, sigma, a * b).reshape(initial_states.shape)
            initial_states = initial_states + s

        fpf = FixedPointFinderTF2(idp.rnn_cell, lr_init=config_optim['lr_init'],
                                  max_iters=config_optim['max_iters'],
                                  do_exclude_distance_outliers = config_optim['exclude_outliers'],
                                  method=config_optim['method'], verbose=1)
        print(f"le he dicho que: {fpf.do_exclude_distance_outliers}")
        unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs,
                                                    cond_ids[config_optim['states_per_chunk'] * i:config_optim['states_per_chunk'] * (i + 1)])
        alls.append(all_fps)
        uniques.append(unique_fps)
        
    alls = FixedPoints.concatenate(alls) 
    uniques = FixedPoints.concatenate(uniques)
 
    # force jacobians and stability in alls
    fpfx = FixedPointFinderTF2(idp.rnn_cell, lr_init=config_optim['lr_init'])
    J_np = fpfx._compute_recurrent_jacobians(alls)
    alls.J_xstar = J_np
    alls.decompose_jacobians()

    if SAVE:
        isExist = os.path.exists(FPS_DIR)
        if not isExist:
            os.makedirs(FPS_DIR)
        alls.save(f'{FPS_DIR}/alls_{file_text}.fps')
        uniques.save(f'{FPS_DIR}/uniques_{file_text}.fps')

    return alls, uniques


def add_np_array(df: pd.DataFrame, data: np.ndarray, col_name: str):
    df_tmp = pd.DataFrame(data)
    df_tmp[col_name] = df_tmp.apply(lambda x: tuple(x), axis=1).apply(np.array)
    df_tmp = df_tmp.drop(df_tmp.columns[:-1], axis=1)
    df = pd.concat([df, df_tmp], axis=1)
    return df

def eigenval_stability(eigenval):
    if eigenval.imag == 0:
        if 0.99 <= abs(eigenval.real) <= 1.01:
            return 'neutral'
        elif abs(eigenval.real) < 0.99:
            return 'stable'
        else:
            return 'unstable'
    else:
        if 0.99 <= abs(eigenval) <= 1.01:
            return 'neutral_center'
        elif abs(eigenval) < 0.99:
            return 'stable_spiral_focus'
        else:
            return 'unstable_spiral_focus'

def fp_stability(eigenvals):
    modes_stabilities = [eigenval_stability(eig) for eig in eigenvals]
    num_inest = sum(map(lambda x: 'unstable' in x, modes_stabilities))
    if len(eigenvals) == num_inest:
        return 'unstable'
    elif num_inest != 0:
        return f'saddle_{num_inest}'
    else:
        return modes_stabilities[0]
    
def color_2_fp(x):
    if 'stable' in x:
        return "green" 
    elif 'neutral' in x:
        return 'fuchsia' 
    elif 'saddle_1' in x:
        return 'red'
    elif 'saddle_2' in x:
        return 'blue'
    else:
        return 'olive'


def fp_to_df(fp):
    df = pd.DataFrame({'qstar': fp.qstar})
    df = add_np_array(df, data=fp.xstar, col_name='xstar')
    df = add_np_array(df, data=fp.eigval_J_xstar, col_name='eigenvals')
    df['cond_id'] = fp.cond_id
    df['is_stable'] = fp.is_stable
    df['estabilidad'] = df['eigenvals'].apply(fp_stability)
    df['plot_color'] = df['estabilidad'].apply(color_2_fp)
    return df