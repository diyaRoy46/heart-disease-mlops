# Model comparison (held-out test set)

|                     |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   cv_roc_auc_mean |   cv_roc_auc_std |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|------------------:|-----------------:|
| logistic_regression |          0.8852 |           0.8387 |        0.9286 |    0.8814 |         0.9654 |            0.9028 |           0.0169 |
| random_forest       |          0.8525 |           0.8065 |        0.8929 |    0.8475 |         0.9513 |            0.9012 |           0.0302 |
| gradient_boosting   |          0.9016 |           0.8667 |        0.9286 |    0.8966 |         0.961  |            0.8744 |           0.0249 |

Exported best model: **logistic_regression** -> `model.joblib`
