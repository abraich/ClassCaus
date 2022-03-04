import streamlit as st
import numpy as np
from utils import *
from main2 import *
from simulation import *
import SessionState
# ignore warnings
import warnings
warnings.filterwarnings("ignore")


st.sidebar.header("Tasks")
tasks_choices = ['Simulation',  'Benchmarking']
task = st.sidebar.selectbox("Choose a task", tasks_choices)

st.title(f"ClassCaus- {task}")


param_sim = {
    'n_features': 25,
    'n_classes': 2,
    'n_samples': 1000,
    'wd_para': 0.,
    'beta': [0.1, 0.1, 0.3],
    'coef_tt': 2.8,
    'rho': 0.1,
    'path_data': './dataclassif.csv'
}

if task == 'Simulation':
    n_samples = st.sidebar.number_input(
        "n_samples", min_value=1000, max_value=10000)
    n_features = st.sidebar.number_input(
        "n_features", min_value=2, max_value=30, value=25)
    coef_tt = st.sidebar.number_input("coef_tt", value=2.8)
    wd_param = st.sidebar.number_input("wd_param", value=0.)

    param_sim['n_samples'] = n_samples
    param_sim['n_features'] = n_features
    param_sim['coef_tt'] = coef_tt
    param_sim['wd_para'] = wd_param

    idx = np.arange(param_sim['n_features'])
    param_sim['beta'] = (-1) ** idx * np.exp(-idx / 20.)

    sim = Simulation(param_sim)
    sim.simule()

    st.write({"Shape": sim.data_sim.shape,
              'WD': sim.wd, "% treatement": sim.perc_treatement})
    # print head datasim
    st.write(sim.data_sim.head())

    # print path
    st.write(f"Data saved in {sim.path_data}")

if task == 'Benchmarking':
    list_models = st.sidebar.multiselect('List of models', [
                                         "ClassCaus", "lgbm", "xgb", "rf", "knn", "mlp", "dt", "lgr"], default=["ClassCaus"])

    params_classifcaus = {
        "encoded_features": 25,
        "alpha_wass": 0.01,
        "batch_size": 128,
        "epochs": 30,
        "lr": 0.001,
        "patience": 10,
    }
    encoded_features = st.sidebar.number_input("encoded_features", value=25)
    alpha_wass = st.sidebar.number_input("alpha_wass", value=0.01)
    batch_size = st.sidebar.number_input("batch_size", value=128)
    epochs = st.sidebar.number_input("epochs", value=30)
    patience = st.sidebar.number_input("patience", value=10)
    N = st.sidebar.number_input("N_test", value=200)

    if st.sidebar.button("Run"):
        Bench = BenchmarkClassif(list_models)

        # Start training
        st.write("Start training")
        # progress bar
        progress = st.progress(0)
        df_results, dic_fig, dic_report = Bench.evall_all_bench(
            params_classifcaus, N)
        progress.progress(100)
        # end training
        st.write("End training")

        # display results
        st.write(df_results)
        # plot from df_results columns acc_0, acc_1, kl_0, kl_1, pehe in function of list_models
        fig_acc = plt.figure(figsize=(10, 10))

        plt.plot(df_results['acc_0'], label='acc_0')
        plt.plot(df_results['acc_1'], label='acc_1')
        plt.plot(df_results['auc_0'], label='auc_0')
        plt.plot(df_results['auc_1'], label='auc_1')
        plt.legend()
        plt.close(fig_acc)
        st.pyplot(fig_acc)

        fig_pehe = plt.figure(figsize=(10, 10))
        plt.plot(df_results['pehe'], label='pehe')
        plt.legend()
        plt.close(fig_pehe)
        st.pyplot(fig_pehe)
        # plot figures
        for key, fig in dic_fig.items():
            # write key as title
            st.subheader(key)
            st.pyplot(fig[0])
            st.pyplot(fig[1])
            st.pyplot(fig[2])
