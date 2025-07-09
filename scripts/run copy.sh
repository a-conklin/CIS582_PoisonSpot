#!/usr/bin/env bash
set -euo pipefail

configs=(
    # --- HT ----------------------------------------------------------
    # configs/config_ht_cifar_10.yaml 
    # configs/config_ht_imagenet.yaml
    # configs/config_ht_slt_10.yaml

    # # --- LC ----------------------------------------------------------
    # configs/config_lc_cifar_10.yaml     # p_tgt = 10 p_sus = 10
    # configs/config_lc_cifar_10_10_25.yaml     # p_tgt = 10 p_sus = 25
    # configs/config_lc_cifar_10_10_50.yaml   # p_tgt = 10 p_sus = 50
    # configs/config_lc_cifar_10_10_75.yaml  # p_tgt = 10 p_sus = 75
    # configs/config_lc_cifar_10_10_100.yaml  # p_tgt = 10 p_sus = 100
    # configs/config_lc_cifar_10_20_50.yaml # p_tgt = 20 p_sus = 50
    # configs/config_lc_cifar_10_30_50.yaml # p_tgt = 30 p_sus = 50
    # configs/config_lc_cifar_10_40_50.yaml # p_tgt = 40 p_sus = 50
    # configs/config_lc_cifar_10_50_50.yaml   # p_tgt = 50 p_sus = 50
    # configs/config_lc_cifar_10_75_75.yaml   # p_tgt = 75 p_sus = 75
    # configs/config_lc_cifar_10_100_100.yaml   # p_tgt = 100 p_sus = 100

    # # --- Mixed -------------------------------------------------------
    # configs/config_mixed_lc_narcissus_cifar_10.yaml     # p_tgt = 10 for lc p_tgt 0.5 for narcissus, p_sus = 50 
    # configs/config_mixed_lc_sa_narcissus_cifar_10.yaml  # p_tgt = 10 for lc p_tgt 0.5 for narcissus, p_sus = 50

    # --- Narcissus ---------------------------------------------------
    configs/config_narcissus_cifar_10.yaml    # p_tgt = 10 p_sus = 10
    configs/config_narcissus_cifar_10_10_25.yaml   # p_tgt = 10 p_sus = 25
    configs/config_narcissus_cifar_10_10_50.yaml  # p_tgt = 10 p_sus = 50
    configs/config_narcissus_cifar_10_10_75.yaml    # p_tgt = 10 p_sus = 75
    configs/config_narcissus_cifar_10_10_100.yaml  # p_tgt = 10 p_sus = 100
    configs/config_narcissus_cifar_10_20_50.yaml  # p_tgt = 20 p_sus = 50
    configs/config_narcissus_cifar_10_30_50.yaml # p_tgt = 30 p_sus = 50
    configs/config_narcissus_cifar_10_40_50.yaml # p_tgt = 40 p_sus = 50
    configs/config_narcissus_cifar_10_50_50.yaml  # p_tgt = 50 p_sus = 50
    configs/config_narcissus_cifar_10_75_75.yaml  # p_tgt = 75 p_sus = 75
    configs/config_narcissus_cifar_10_100_100.yaml  # p_tgt = 100 p_sus = 100

    # --- SA ----------------------------------------------------------
    configs/config_sa_cifar_10.yaml      # p_tgt = 10 p_sus = 10
    configs/config_sa_cifar_10_10_25.yaml  # p_tgt = 10 p_sus = 25
    configs/config_sa_cifar_10_10_50.yaml # p_tgt = 10 p_sus = 50
    configs/config_sa_cifar_10_10_75.yaml # p_tgt = 10 p_sus = 75
    configs/config_sa_cifar_10_10_100.yaml # p_tgt = 10 p_sus = 100
    configs/config_sa_cifar_10_20_50.yaml # p_tgt = 20 p_sus = 50
    configs/config_sa_cifar_10_30_50.yaml # p_tgt = 30 p_sus = 50
    configs/config_sa_cifar_10_random.yaml # p_tgt = 10 p_sus = 50
    configs/config_sa_cifar_10_fine_tune.yaml # p_tgt = 10 p_sus = 50
    configs/config_sa_slt_10.yaml 
)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for cfg in "${configs[@]}"; do
    echo "============================================================"
    echo " Running experiment with: ${cfg}"
    echo "============================================================"
    python3 main.py --config "${cfg}"
done

echo "All done!"
