from electrodialysis_experiment.utils.hdf_viewing import *
import pandas as pd

def main():
    h5_path = "src/electrodialysis_experiment/data/output/m_concSSE_minimized_slkocvcu_with_surrloglin_cc_init1_cuun.h5"
    res = search_components(
    h5_path,
    kind="variable",
    comp_keywords=["sample_blk[0]", "ion_trans_number_membrane"],
    index_keywords=["Na_+", "cem"],
       # include value (and always component/index)
)
    print(res.case, res.n_rows, res.n_components)
    found=res.matches
    print(found)
    print(found["component"].unique())
    print(found["index"].unique())




if __name__ == "__main__":
    main()
