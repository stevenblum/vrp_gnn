import pandas as pd
from FacExp import FacExp

def main():
    # Example: 3 factors, 2 levels each
    grid = {
        "weight_decay": [10**(-6),10**(-5)],
        "embedding_dim": [256, 512],
        "num_encoder_layers": [5, 7],
        "num_attn_heads": [8, 16],
        "vehicle_capacity": [120,150],
        "num_train_locs": [100, 130],
        "train_set_clusters": [4, 8],
    }

    # Defining contrast: C = AB (i.e., keep only rows where the product of A and B and C is +1)
    contrasts = ['ABC',"DEF"]
    exp = FacExp(grid, defining_contrasts=contrasts)
    df = exp.get_exp_combinations()
    print("Fractional factorial design (levels):")
    print(df)

    print("\nFractional factorial design (+/-):")
    print(exp.get_exp_sign_table())

if __name__ == "__main__":
    main()
