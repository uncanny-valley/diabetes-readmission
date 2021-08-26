python3 src/models/tuning.py --model logreg --scoring recall --num_trials 1000 --dataset original --early-stopping 100 --study-name logreg_original_recall
python3 src/models/tuning.py --model logreg --scoring f2 --num_trials 1000 --dataset original --early-stopping 100 --study-name logreg_original_f2
python3 src/models/tuning.py --model logreg --scoring roc_auc --num_trials 1000 --dataset original --early-stopping 100 --study-name logreg_original_roc_auc
python3 src/models/tuning.py --model logreg --scoring f1 --num_trials 1000 --dataset original --early-stopping 100 --study-name logreg_original_f1

python3 src/models/tuning.py --model logreg --scoring recall --num_trials 1000 --dataset rfecv --early-stopping 100 --study-name logreg_rfecv_recall
python3 src/models/tuning.py --model logreg --scoring f2 --num_trials 1000 --dataset rfecv --early-stopping 100 --study-name logreg_rfecv_f2
python3 src/models/tuning.py --model logreg --scoring roc_auc --num_trials 1000 --dataset rfecv --early-stopping 100 --study-name logreg_rfecv_roc_auc
python3 src/models/tuning.py --model logreg --scoring f1 --num_trials 1000 --dataset rfecv --early-stopping 100 --study-name logreg_rfecv_f1

python3 src/models/tuning.py --model logreg --scoring recall --num_trials 1000 --dataset pruned --early-stopping 100 --study-name logreg_pruned_recall
python3 src/models/tuning.py --model logreg --scoring f2 --num_trials 1000 --dataset pruned --early-stopping 100 --study-name logreg_pruned_f2
python3 src/models/tuning.py --model logreg --scoring roc_auc --num_trials 1000 --dataset pruned --early-stopping 100 --study-name logreg_pruned_roc_auc
python3 src/models/tuning.py --model logreg --scoring f1 --num_trials 1000 --dataset pruned --early-stopping 100 --study-name logreg_pruned_f1

python3 src/models/tuning.py --model logreg --scoring recall --num_trials 1000 --dataset top_35 --early-stopping 100 --study-name logreg_top_35_recall
python3 src/models/tuning.py --model logreg --scoring f2 --num_trials 1000 --dataset top_35 --early-stopping 100 --study-name logreg_top_35_f2
python3 src/models/tuning.py --model logreg --scoring roc_auc --num_trials 1000 --dataset top_35 --early-stopping 100 --study-name logreg_top_35_roc_auc
python3 src/models/tuning.py --model logreg --scoring f1 --num_trials 1000 --dataset top_35 --early-stopping 100 --study-name logreg_top_35_f1


python3 src/models/tuning.py --model random_forest --scoring recall --num_trials 1000 --dataset original --early-stopping 200 --study-name random_forest_original_recall
python3 src/models/tuning.py --model random_forest --scoring f2 --num_trials 1000 --dataset original --early-stopping 200 --study-name random_forest_original_f2
python3 src/models/tuning.py --model random_forest --scoring roc_auc --num_trials 1000 --dataset original --early-stopping 200 --study-name random_forest_original_roc_auc
python3 src/models/tuning.py --model random_forest --scoring f1 --num_trials 1000 --dataset original --early-stopping 200 --study-name random_forest_original_f1

python3 src/models/tuning.py --model random_forest --scoring recall --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name random_forest_rfecv_recall
python3 src/models/tuning.py --model random_forest --scoring f2 --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name random_forest_rfecv_f2
python3 src/models/tuning.py --model random_forest --scoring roc_auc --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name random_forest_rfecv_roc_auc
python3 src/models/tuning.py --model random_forest --scoring f1 --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name random_forest_rfecv_f1

python3 src/models/tuning.py --model random_forest --scoring recall --num_trials 1000 --dataset pruned --early-stopping 200 --study-name random_forest_pruned_recall
python3 src/models/tuning.py --model random_forest --scoring f2 --num_trials 1000 --dataset pruned --early-stopping 200 --study-name random_forest_pruned_f2
python3 src/models/tuning.py --model random_forest --scoring roc_auc --num_trials 1000 --dataset pruned --early-stopping 200 --study-name random_forest_pruned_roc_auc
python3 src/models/tuning.py --model random_forest --scoring f1 --num_trials 1000 --dataset pruned --early-stopping 200 --study-name random_forest_pruned_f1

python3 src/models/tuning.py --model random_forest --scoring recall --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name random_forest_top_35_recall
python3 src/models/tuning.py --model random_forest --scoring f2 --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name random_forest_top_35_f2
python3 src/models/tuning.py --model random_forest --scoring roc_auc --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name random_forest_top_35_roc_auc
python3 src/models/tuning.py --model random_forest --scoring f1 --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name random_forest_top_35_f1


python3 src/models/tuning.py --model lgbm --scoring recall --num_trials 1000 --dataset original --early-stopping 200 --study-name lgbm_original_recall
python3 src/models/tuning.py --model lgbm --scoring f2 --num_trials 1000 --dataset original --early-stopping 200 --study-name lgbm_original_f2
python3 src/models/tuning.py --model lgbm --scoring roc_auc --num_trials 1000 --dataset original --early-stopping 200 --study-name lgbm_original_roc_auc
python3 src/models/tuning.py --model lgbm --scoring f1 --num_trials 1000 --dataset original --early-stopping 200 --study-name lgbm_original_f1

python3 src/models/tuning.py --model lgbm --scoring recall --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name lgbm_rfecv_recall
python3 src/models/tuning.py --model lgbm --scoring f2 --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name lgbm_rfecv_f2
python3 src/models/tuning.py --model lgbm --scoring roc_auc --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name lgbm_rfecv_roc_auc
python3 src/models/tuning.py --model lgbm --scoring f1 --num_trials 1000 --dataset rfecv --early-stopping 200 --study-name lgbm_rfecv_f1

python3 src/models/tuning.py --model lgbm --scoring recall --num_trials 1000 --dataset pruned --early-stopping 200 --study-name lgbm_pruned_recall
python3 src/models/tuning.py --model lgbm --scoring f2 --num_trials 1000 --dataset pruned --early-stopping 200 --study-name lgbm_pruned_f2
python3 src/models/tuning.py --model lgbm --scoring roc_auc --num_trials 1000 --dataset pruned --early-stopping 200 --study-name lgbm_pruned_roc_auc
python3 src/models/tuning.py --model lgbm --scoring f1 --num_trials 1000 --dataset pruned --early-stopping 200 --study-name lgbm_pruned_f1

python3 src/models/tuning.py --model lgbm --scoring recall --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name lgbm_top_35_recall
python3 src/models/tuning.py --model lgbm --scoring f2 --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name lgbm_top_35_f2
python3 src/models/tuning.py --model lgbm --scoring roc_auc --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name lgbm_top_35_roc_auc
python3 src/models/tuning.py --model lgbm --scoring f1 --num_trials 1000 --dataset top_35 --early-stopping 200 --study-name lgbm_top_35_f1

