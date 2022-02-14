import streamlit as st
import pandas as pd
import numpy as np
import pulp as pl
import pickle
import time

# read data
from optimizer import roles, waku_dict, new_optimizer, list_objectives

# read data
level_dict = pickle.load(open("notebook/level-dqw.pickle", "rb"))

# session state
if "constraints" not in st.session_state:
    st.session_state["constraints"] = set({})

if "prev_result" not in st.session_state:
    st.session_state["prev_result"] = []


def main():
    # app
    st.title("DQW Kokoroptimizer")

    # choice
    option = st.selectbox("職業", roles)
    # waku_list = waku_dict[option]

    level = st.slider("レベル", min_value=1, max_value=79, value=55)

    st.subheader(f"Optimization for {option} | max cost {level_dict['joukyu'][level]}")

    opt_option = st.selectbox("目的関数", list_objectives)

    def form_callback():
        for n in range(4):
            key = f"selection{n}"
            if st.session_state[key] == "No":
                prev_result = st.session_state["prev_result"]
                st.session_state["constraints"].add(prev_result[n][-1])
            # initialize toggle state
            st.session_state[key] = "Yes"

    # 結果を作成
    # st.write(st.session_state["constraints"])
    with st.spinner("Wait for optimization") as f:
        result = new_optimizer(
            syokugyou=option,
            max_cost=level_dict["joukyu"][level],
            constraints=st.session_state["constraints"],
        )
        st.session_state["prev_result"] = result

    with st.form(key="result_form"):
        cols = st.columns(4)
        for (idc, col) in enumerate(cols):
            col.write(f"こころ{idc + 1}")
            # col.write(f"{waku_list[idc]}")
            col.write(result[idc][0])
            cv = col.selectbox(f"Have?", ["Yes", "No"], key=f"selection{idc}")
            if cv == "No":
                st.session_state["constraints"].add(result[idc][-1])
                # st.session_state["selection{idc}"] = "Yes"

        st.form_submit_button(label="Re-optimize", on_click=form_callback)


if __name__ == "__main__":
    main()
